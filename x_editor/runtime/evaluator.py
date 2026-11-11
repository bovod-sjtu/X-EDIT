"""Evaluator interfaces for structured audio-edit execution."""

from __future__ import annotations

import base64
import asyncio
import json
import mimetypes
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

from x_editor.agents.base import GeminiClient
from x_editor.core.config import AudioEditConfig, resolve_gemini_api_key, runtime_artifact_path
from x_editor.core.schemas import AudioEditPlan, EvaluationResult


DEFAULT_QWEN_OMNI_MODEL = "Qwen3Omni-Instruct"
DEFAULT_GEMINI_EVALUATOR_MODEL = "gemini-2.5-pro"
DEFAULT_PASSING_SCORE = 0.75
DEFAULT_MAX_RUBRICS = 5
DEFAULT_EVALUATOR_TOOL_IDS = {
    "audio_effects",
    "edit_data_pipeline",
    "speech_enhancement",
    "speech_separation",
    "tse",
}
DEFAULT_EVALUATOR_RUNTIME_OPERATIONS = {
    "artifact_ops.extract_spans",
    "perception.speech_context",
    "perception.audio_fallback",
    "perception.gemini_audio",
}


class Evaluator(Protocol):
    """Evaluation interface reserved for future perceptual and task checks."""

    def evaluate(
        self,
        plan: AudioEditPlan,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        ...


class StubEvaluator:
    """Minimal evaluator based only on step statuses.

    Future implementations can add audio-quality checks, ASR/content checks,
    loudness validation, target-speaker checks, user-goal consistency checks,
    and LLM-as-a-judge review.
    """

    def evaluate(
        self,
        plan: AudioEditPlan,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        failed_steps = [step.id for step in plan.steps if step.required and step.status == "failed"]
        pending_steps = [step.id for step in plan.steps if step.required and step.status in {"pending", "running"}]
        skipped_steps = [step.id for step in plan.steps if step.required and step.status == "skipped"]
        blocking = failed_steps + pending_steps + skipped_steps
        passed = not blocking
        if passed:
            feedback = "All required audio edit steps completed successfully."
        else:
            feedback = "Some required audio edit steps did not complete: " + ", ".join(blocking)
        return EvaluationResult(
            passed=passed,
            score=None,
            feedback=feedback,
            need_replan=False,
            failed_steps=blocking,
            metadata={
                "failed_steps": failed_steps,
                "pending_steps": pending_steps,
                "skipped_steps": skipped_steps,
            },
        )


@dataclass
class VerificationRubric:
    """One audio-grounded yes/no verification question.

    ``pass_choice`` is intentionally simple.  The current Qwen3-Omni judge asks
    every rubric as ``A=Yes/pass, B=No/fail, C=Cannot determine`` so rubrics
    remain easy to aggregate outside MMAE.
    """

    rubric_id: str
    category: str
    question: str
    weight: float = 1.0
    required: bool = True
    pass_choice: str = "A"
    evidence_source: str = "instruction_only"
    choices: list[str] = field(
        default_factory=lambda: [
            "Yes, the edited audio satisfies this requirement",
            "No, the edited audio does not satisfy this requirement",
            "Cannot determine from the available audio",
        ]
    )
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "category": self.category,
            "question": self.question,
            "weight": self.weight,
            "required": self.required,
            "pass_choice": self.pass_choice,
            "evidence_source": self.evidence_source,
            "choices": self.choices,
            "rationale": self.rationale,
        }


class Qwen3OmniJudgeClient:
    """Small OpenAI-compatible Qwen3-Omni judge client.

    The local MMAE scorer uses the same message shape: audio files are embedded
    as ``audio_url`` data URIs and sent to ``/chat/completions``.
    """

    def __init__(
        self,
        *,
        base_urls: list[str] | tuple[str, ...] | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        service_bootstrap_agent: Any | None = None,
        enable_service_bootstrap: bool | None = None,
    ) -> None:
        self.base_urls = list(base_urls or _qwen_base_urls_from_env() or _qwen_default_base_urls_when_bootstrap_configured())
        self.model = model or os.getenv("X_EDITOR_QWEN_OMNI_EVAL_MODEL", DEFAULT_QWEN_OMNI_MODEL)
        self.api_key = api_key or _qwen_api_key_from_env()
        self.timeout = float(timeout or os.getenv("X_EDITOR_QWEN_OMNI_EVAL_TIMEOUT", "300"))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.service_bootstrap_agent = service_bootstrap_agent
        self.enable_service_bootstrap = enable_service_bootstrap
        self._index = 0

    @property
    def available(self) -> bool:
        return bool(self.base_urls)

    @property
    def unavailable_reason(self) -> str:
        return "no Qwen3-Omni evaluator base URL is configured"

    def ensure_service(self, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Optionally start the OpenAI-compatible Qwen3-Omni judge service."""

        context = context or {}
        specs = _qwen_evaluator_service_specs(context, self.base_urls)
        if not specs:
            return None
        if not self.base_urls:
            self.base_urls = _base_urls_from_service_specs(specs)
        from x_editor.tools.service_bootstrap import ServiceBootstrapAgent

        enabled = self.enable_service_bootstrap
        if enabled is None:
            enabled = _context_bool_optional(
                context,
                "enable_evaluator_service_bootstrap",
                env_name="X_EDITOR_ENABLE_EVALUATOR_SERVICE_BOOTSTRAP",
            )
        agent = self.service_bootstrap_agent or ServiceBootstrapAgent(
            enabled=enabled,
        )
        report = _run_coroutine_sync(
            agent.ensure_tool_services(
                tool_id="qwen3_omni_evaluator",
                tool_dir=Path.cwd(),
                tool_config={"managed_services": specs},
                arguments={},
            )
        )
        return report.to_dict()

    def call_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        audio_paths: list[str] | None = None,
        audio_labels: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.base_urls:
            raise RuntimeError(
                "Qwen3-Omni evaluator has no base URL. Set "
                "X_EDITOR_QWEN_OMNI_EVAL_BASE_URLS or QWEN3_OMNI_BASE_URLS."
            )
        text = self.call(
            system_prompt=system_prompt,
            user_text=user_text,
            audio_paths=audio_paths,
            audio_labels=audio_labels,
        )
        parsed = _parse_json_object(text)
        if parsed is None:
            raise RuntimeError(f"Qwen3-Omni judge returned non-JSON response: {text[:1000]}")
        return parsed

    def call(
        self,
        *,
        system_prompt: str,
        user_text: str,
        audio_paths: list[str] | None = None,
        audio_labels: list[str] | None = None,
    ) -> str:
        url = self._next_url()
        content: list[dict[str, Any]] = []
        for index, audio_path in enumerate(audio_paths or []):
            label = (audio_labels or [])[index] if index < len(audio_labels or []) else f"Audio {index + 1}"
            content.append({"type": "text", "text": f"{label}: "})
            content.append({"type": "audio_url", "audio_url": {"url": _audio_data_uri(audio_path)}})
        content.append({"type": "text", "text": user_text})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Qwen3-Omni judge status {response.status_code}: {response.text[:800]}")
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices:
            raise RuntimeError(f"Qwen3-Omni judge returned no choices: {body}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        content_text = message.get("content") if isinstance(message, dict) else None
        if isinstance(content_text, list):
            content_text = "".join(str(part.get("text", "")) for part in content_text if isinstance(part, dict))
        if not isinstance(content_text, str) or not content_text.strip():
            raise RuntimeError(f"Qwen3-Omni judge returned empty content: {body}")
        return content_text.strip()

    def _next_url(self) -> str:
        value = self.base_urls[self._index % len(self.base_urls)]
        self._index += 1
        return value


class GeminiAudioJudgeClient:
    """Gemini 2.5 Pro audio judge client using the existing Gemini endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> None:
        config = AudioEditConfig()
        self.model = model or os.getenv("X_EDITOR_GEMINI_EVAL_MODEL", DEFAULT_GEMINI_EVALUATOR_MODEL)
        self.api_key = resolve_gemini_api_key(api_key) or config.omni_api_key
        self.client = GeminiClient(
            api_key=self.api_key,
            base_url=_format_model_base_url(
                base_url or os.getenv("X_EDITOR_GEMINI_EVAL_BASE_URL") or config.omni_base_url,
                self.model,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(timeout or os.getenv("X_EDITOR_GEMINI_EVAL_TIMEOUT", "120")),
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def unavailable_reason(self) -> str:
        return "no Gemini API key is configured"

    def call_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        audio_paths: list[str] | None = None,
        audio_labels: list[str] | None = None,
    ) -> dict[str, Any]:
        label_text = "\n".join(
            f"Audio {index + 1}: {(audio_labels or [])[index] if index < len(audio_labels or []) else Path(path).name}"
            for index, path in enumerate(audio_paths or [])
        )
        if label_text:
            user_text = f"{label_text}\n\n{user_text}"
        return self.client.call_json(system_prompt, user_text, audio_paths=list(audio_paths or []))


class GeminiEvaluatorToolCallPlanner:
    """Gemini helper that proposes evidence-only tool/runtime calls for evaluation."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 8192,
        timeout: float = 60.0,
    ) -> None:
        config = AudioEditConfig()
        self.model_name = model_name or os.getenv("X_EDITOR_GEMINI_EVALUATOR_TOOL_MODEL", "gemini-2.5-pro")
        self.api_key = resolve_gemini_api_key(api_key) or config.omni_api_key
        self.client = GeminiClient(
            api_key=self.api_key,
            base_url=base_url or os.getenv("X_EDITOR_GEMINI_EVALUATOR_TOOL_BASE_URL") or config.omni_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def plan_evaluation_tool_calls(
        self,
        *,
        plan: AudioEditPlan,
        context: dict[str, Any],
        allowed_tool_ids: set[str],
        allowed_runtime_operations: set[str],
    ) -> dict[str, list[dict[str, Any]]]:
        if not self.api_key:
            raise RuntimeError("Shared Gemini 2.5 Pro API key is required for evaluator tool planning")
        prompt = _gemini_evaluator_tool_prompt(
            plan=plan,
            context=context,
            model_name=self.model_name,
            allowed_tool_ids=allowed_tool_ids,
            allowed_runtime_operations=allowed_runtime_operations,
        )
        audio_paths = []
        if _context_bool(
            context,
            "evaluator_tool_planner_audio",
            env_name="X_EDITOR_EVALUATOR_TOOL_PLANNER_AUDIO",
        ):
            input_audio = context.get("input_audio") or plan.input_audio
            output_audio = _final_audio_path(plan)
            for path in [input_audio, output_audio]:
                if path and Path(str(path)).exists():
                    audio_paths.append(str(path))
        data = self.client.call_json(
            _GEMINI_EVALUATOR_TOOL_SYSTEM_PROMPT,
            prompt,
            audio_paths=audio_paths,
        )
        return _validated_gemini_evaluator_tool_calls(
            data,
            allowed_tool_ids=allowed_tool_ids,
            allowed_runtime_operations=allowed_runtime_operations,
        )


class EvaluatorToolEvidenceProvider:
    """Collect verifier evidence by calling existing X-Editor tools/runtime operations.

    Direct catalog calls go through ``ToolExecutor``, so they inherit service
    bootstrap, health checks, smoke tests, and MCP invocation behavior from the
    executor path. Runtime steps go through ``AudioEditExecutor`` in a separate
    evidence output directory, so perception/artifact operations can be reused
    without altering the edited audio being judged.
    """

    def __init__(
        self,
        *,
        tool_executor: Any | None = None,
        audio_executor: Any | None = None,
        gemini_tool_planner: GeminiEvaluatorToolCallPlanner | Any | None = None,
        enable_gemini_tools: bool | None = None,
        allowed_tool_ids: set[str] | None = None,
        allowed_runtime_operations: set[str] | None = None,
    ) -> None:
        self.tool_executor = tool_executor
        self.audio_executor = audio_executor
        self.gemini_tool_planner = gemini_tool_planner
        self.enable_gemini_tools = enable_gemini_tools
        self.allowed_tool_ids = allowed_tool_ids
        self.allowed_runtime_operations = allowed_runtime_operations

    def collect_evaluation_evidence(
        self,
        *,
        plan: AudioEditPlan,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        output_dir = Path(
            context.get("evaluation_tool_output_dir")
            or context.get("evaluator_tool_output_dir")
            or Path(context.get("output_dir") or runtime_artifact_path()) / "evaluator_tools"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        allowed_tool_ids = _allowed_evaluator_tool_ids(context, self.allowed_tool_ids)
        allowed_runtime_operations = _allowed_evaluator_runtime_operations(
            context,
            self.allowed_runtime_operations,
        )
        tool_calls = _coerce_dict_list(context.get("evaluation_tool_calls"))
        runtime_steps = _coerce_dict_list(context.get("evaluation_runtime_steps"))
        errors: list[str] = []

        if _enable_gemini_evaluator_tools(context, self.enable_gemini_tools):
            planner = self.gemini_tool_planner or GeminiEvaluatorToolCallPlanner()
            try:
                planned = planner.plan_evaluation_tool_calls(
                    plan=plan,
                    context=context,
                    allowed_tool_ids=allowed_tool_ids,
                    allowed_runtime_operations=allowed_runtime_operations,
                )
                tool_calls.extend(planned.get("tool_calls") or [])
                runtime_steps.extend(planned.get("runtime_steps") or [])
            except Exception as exc:
                errors.append(f"gemini_evaluator_tool_planner: {type(exc).__name__}: {exc}")

        evidence: dict[str, Any] = {
            "source": "evaluator_tool_evidence_provider",
            "trusted_time_spans": [],
            "tool_results": [],
            "runtime_results": [],
            "service_bootstrap": [],
            "errors": errors,
        }
        if tool_calls:
            self._execute_tool_calls(
                plan=plan,
                context=context,
                output_dir=output_dir,
                calls=tool_calls,
                allowed_tool_ids=allowed_tool_ids,
                evidence=evidence,
            )
        if runtime_steps:
            self._execute_runtime_steps(
                plan=plan,
                context=context,
                output_dir=output_dir,
                steps=runtime_steps,
                allowed_runtime_operations=allowed_runtime_operations,
                evidence=evidence,
            )
        return _without_empty(evidence)

    def _execute_tool_calls(
        self,
        *,
        plan: AudioEditPlan,
        context: dict[str, Any],
        output_dir: Path,
        calls: list[dict[str, Any]],
        allowed_tool_ids: set[str],
        evidence: dict[str, Any],
    ) -> None:
        from x_editor.tools.executor import ToolExecutor

        executor = self.tool_executor or ToolExecutor(
            enable_service_bootstrap=_context_bool_optional(
                context,
                "enable_evaluator_service_bootstrap",
                env_name="X_EDITOR_ENABLE_EVALUATOR_SERVICE_BOOTSTRAP",
            )
        )
        owns_executor = self.tool_executor is None
        try:
            for index, call in enumerate(calls):
                tool_id = str(call.get("tool_id") or "").strip()
                if tool_id not in allowed_tool_ids:
                    evidence.setdefault("errors", []).append(f"tool_call[{index}] rejected unsupported tool_id={tool_id!r}")
                    continue
                try:
                    arguments = _prepare_evaluator_tool_arguments(
                        tool_id=tool_id,
                        raw_arguments=_mapping(call.get("arguments")),
                        plan=plan,
                        context=context,
                        output_dir=output_dir,
                        index=index,
                    )
                    result = _run_coroutine_sync(executor.execute(tool_id, arguments))
                    output = result.output if isinstance(getattr(result, "output", None), dict) else {}
                    evidence["tool_results"].append(
                        _without_empty(
                            {
                                "tool_id": tool_id,
                                "success": bool(getattr(result, "success", False)),
                                "arguments": _safe_prompt_mapping(arguments),
                                "output": _safe_prompt_value(output, depth=3),
                                "error_message": getattr(result, "error_message", None),
                            }
                        )
                    )
                    if isinstance(output.get("service_bootstrap"), dict):
                        evidence["service_bootstrap"].append(output["service_bootstrap"])
                    for report in getattr(executor, "bootstrap_reports", []) or []:
                        if report not in evidence["service_bootstrap"]:
                            evidence["service_bootstrap"].append(report)
                    evidence["trusted_time_spans"].extend(
                        _trusted_spans_from_tool_output(output, source_prefix=f"tool_result.{tool_id}")
                    )
                except Exception as exc:
                    evidence.setdefault("errors", []).append(f"tool_call[{tool_id}]: {type(exc).__name__}: {exc}")
        finally:
            if owns_executor:
                _run_coroutine_sync(executor.shutdown())

    def _execute_runtime_steps(
        self,
        *,
        plan: AudioEditPlan,
        context: dict[str, Any],
        output_dir: Path,
        steps: list[dict[str, Any]],
        allowed_runtime_operations: set[str],
        evidence: dict[str, Any],
    ) -> None:
        safe_steps = []
        for index, step in enumerate(steps):
            operation = str(step.get("operation") or "").strip()
            if operation not in allowed_runtime_operations:
                evidence.setdefault("errors", []).append(
                    f"runtime_step[{index}] rejected unsupported operation={operation!r}"
                )
                continue
            prepared = dict(step)
            prepared.setdefault("id", f"eval_{index + 1}_{operation.replace('.', '_')}")
            prepared.setdefault("description", "Collect evaluator evidence.")
            prepared.setdefault("inputs", {"input_audio_path": "current"})
            prepared.setdefault("params", {})
            prepared.setdefault("depends_on", [])
            safe_steps.append(prepared)
        if not safe_steps:
            return
        if self.audio_executor is None:
            from x_editor.runtime.audio_plan_executor import AudioEditExecutor

            audio_executor = AudioEditExecutor(output_dir=output_dir)
        else:
            audio_executor = self.audio_executor
        evidence_plan = AudioEditPlan(
            request=f"Collect evaluator evidence for: {plan.request}",
            input_audio=context.get("input_audio") or plan.input_audio,
            output_audio=None,
            steps=safe_steps,
            global_params={
                "source_text": context.get("source_text") or plan.global_params.get("source_text"),
                "perception_report": context.get("perception_report") or plan.global_params.get("perception_report") or {},
                "reference_audio_paths": context.get("reference_audio_paths") or plan.global_params.get("reference_audio_paths") or [],
                "output_dir": str(output_dir),
            },
            metadata={"evaluator_evidence_collection": True},
        )
        executed = _run_coroutine_sync(
            audio_executor.aexecute(
                evidence_plan,
                context={
                    **context,
                    "output_dir": str(output_dir),
                    "output_audio": None,
                },
            )
        )
        evidence["runtime_results"].append(executed.model_dump())
        for step in executed.steps:
            if isinstance(step.result, dict):
                evidence["trusted_time_spans"].extend(
                    _trusted_spans_from_tool_output(
                        step.result,
                        source_prefix=f"step_result.{step.operation}",
                    )
                )
        for report in executed.metadata.get("service_bootstrap") or []:
            evidence["service_bootstrap"].append(report)


class AudioJudgeEvaluator:
    """Audio-grounded evaluator that asks a multimodal judge generated rubrics.

    This evaluator is intentionally benchmark-agnostic. It first asks the judge
    to derive compact rubrics from the user request and executed plan, then asks
    those rubrics over the original and edited audio.
    """

    def __init__(
        self,
        *,
        client: Any,
        passing_score: float | None = None,
        max_rubrics: int | None = None,
        judge_name: str,
        display_name: str,
        rubric_system_prompt: str,
        answer_system_prompt: str,
        tool_evidence_provider: EvaluatorToolEvidenceProvider | Any | None = None,
        tool_executor: Any | None = None,
        audio_executor: Any | None = None,
        gemini_tool_planner: GeminiEvaluatorToolCallPlanner | Any | None = None,
        enable_tools: bool | None = None,
        enable_gemini_tools: bool | None = None,
    ) -> None:
        self.client = client
        self.judge_name = judge_name
        self.display_name = display_name
        self.rubric_system_prompt = rubric_system_prompt
        self.answer_system_prompt = answer_system_prompt
        self.enable_tools = enable_tools
        self.tool_evidence_provider = tool_evidence_provider or EvaluatorToolEvidenceProvider(
            tool_executor=tool_executor,
            audio_executor=audio_executor,
            gemini_tool_planner=gemini_tool_planner,
            enable_gemini_tools=enable_gemini_tools,
        )
        self.passing_score = float(
            passing_score
            if passing_score is not None
            else _float_from_env(
                "X_EDITOR_AUDIO_EVAL_PASSING_SCORE",
                "X_EDITOR_QWEN_OMNI_PASSING_SCORE",
                default=DEFAULT_PASSING_SCORE,
            )
        )
        self.max_rubrics = int(
            max_rubrics
            or _int_from_env(
                "X_EDITOR_AUDIO_EVAL_MAX_RUBRICS",
                "X_EDITOR_QWEN_OMNI_MAX_RUBRICS",
                default=DEFAULT_MAX_RUBRICS,
            )
        )

    def evaluate(
        self,
        plan: AudioEditPlan,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        context = dict(context or {})
        status_result = _status_evaluation(plan)
        output_audio = _final_audio_path(plan)
        input_audio = _existing_path(context.get("input_audio") or plan.input_audio)
        reference_audios = [
            str(path)
            for path in context.get("reference_audio_paths", []) or []
            if _existing_path(path)
        ]
        metadata: dict[str, Any] = {
            "evaluator": self.judge_name,
            "evaluator_display_name": self.display_name,
            "evaluator_model": getattr(self.client, "model", None),
            "status_gate": status_result.metadata,
            "input_audio_path": str(input_audio) if input_audio else None,
            "output_audio_path": str(output_audio) if output_audio else None,
            "reference_audio_paths": reference_audios,
            "passing_score": self.passing_score,
        }

        if not status_result.passed:
            return EvaluationResult(
                passed=False,
                score=0.0,
                feedback=status_result.feedback,
                need_replan=True,
                failed_steps=status_result.failed_steps,
                metadata=metadata,
            )
        if not output_audio or not Path(output_audio).exists():
            return EvaluationResult(
                passed=False,
                score=0.0,
                feedback="Execution completed, but no final output audio exists for verification.",
                need_replan=True,
                failed_steps=["final_output"],
                metadata=metadata,
            )

        service_report = _ensure_judge_service(self.client, context)
        if service_report:
            metadata["judge_service_bootstrap"] = service_report
            if not service_report.get("ok", True):
                feedback = (
                    f"Execution completed, but {self.display_name} verification was not run because the judge "
                    "service could not be started or smoke-tested."
                )
                metadata["verification_unavailable"] = True
                metadata["verification_unavailable_reason"] = _bootstrap_error_message_from_report(service_report)
                return EvaluationResult(
                    passed=False,
                    score=None,
                    feedback=feedback,
                    need_replan=False,
                    metadata=metadata,
                )

        if not getattr(self.client, "available", True):
            reason = getattr(self.client, "unavailable_reason", "the evaluator client is not configured")
            feedback = (
                f"Execution completed, but {self.display_name} verification was not run because no evaluator "
                f"provider is available ({reason}). Output is returned with verification_unavailable=true."
            )
            metadata["verification_unavailable"] = True
            metadata["verification_unavailable_reason"] = reason
            return EvaluationResult(
                passed=False,
                score=None,
                feedback=feedback,
                need_replan=False,
                metadata=metadata,
            )

        try:
            evidence_context = self._context_with_tool_evidence_provider(context)
            rubric_evidence = _rubric_evidence_context(plan, evidence_context)
            metadata["rubric_evidence"] = rubric_evidence
            rubrics = self._generate_rubrics(plan, context, input_audio, output_audio, rubric_evidence)
            answers = [
                self._answer_rubric(rubric, input_audio, output_audio, reference_audios)
                for rubric in rubrics
            ]
        except Exception as exc:
            metadata["verification_error"] = f"{type(exc).__name__}: {exc}"
            return EvaluationResult(
                passed=False,
                score=None,
                feedback=(
                    f"Execution completed, but {self.display_name} verification failed before a reliable "
                    f"judgment could be produced: {type(exc).__name__}: {exc}"
                ),
                need_replan=False,
                metadata=metadata,
            )

        score = _weighted_score(answers)
        failed_required = [
            item for item in answers
            if item.get("required", True) and not item.get("passed", False)
        ]
        passed = score >= self.passing_score and not failed_required
        failed_steps = [str(item["rubric_id"]) for item in answers if not item.get("passed", False)]
        fixable = any(item.get("fixable", True) for item in answers if not item.get("passed", False))
        metadata.update(
            {
                "rubrics": [rubric.to_dict() for rubric in rubrics],
                "answers": answers,
                "failed_required_rubrics": [item["rubric_id"] for item in failed_required],
            }
        )
        return EvaluationResult(
            passed=passed,
            score=score,
            feedback=_evaluation_feedback(passed, score, self.passing_score, answers, self.display_name),
            need_replan=not passed and fixable,
            failed_steps=failed_steps,
            metadata=metadata,
        )

    def _context_with_tool_evidence_provider(self, context: dict[str, Any]) -> dict[str, Any]:
        if not _evaluator_tooling_requested(context, self.enable_tools):
            return context
        updated = dict(context)
        providers = []
        existing = updated.get("evaluation_evidence_providers")
        if isinstance(existing, list):
            providers.extend(existing)
        elif existing is not None:
            providers.append(existing)
        providers.append(self.tool_evidence_provider)
        updated["evaluation_evidence_providers"] = providers
        return updated

    def _generate_rubrics(
        self,
        plan: AudioEditPlan,
        context: dict[str, Any],
        input_audio: Path | None,
        output_audio: str | Path,
        rubric_evidence: dict[str, Any],
    ) -> list[VerificationRubric]:
        prompt = _rubric_generation_prompt(
            plan,
            context,
            bool(input_audio),
            bool(output_audio),
            rubric_evidence,
        )
        data = self.client.call_json(
            system_prompt=self.rubric_system_prompt,
            user_text=prompt,
        )
        rubrics = _rubrics_from_model(
            data,
            max_rubrics=self.max_rubrics,
            trusted_time_spans=rubric_evidence.get("trusted_time_spans") or [],
        )
        if rubrics:
            return rubrics
        return _fallback_rubrics(plan.request, has_input_audio=bool(input_audio), max_rubrics=self.max_rubrics)

    def _answer_rubric(
        self,
        rubric: VerificationRubric,
        input_audio: Path | None,
        output_audio: str | Path,
        reference_audios: list[str],
    ) -> dict[str, Any]:
        audio_paths = []
        audio_labels = []
        if input_audio:
            audio_paths.append(str(input_audio))
            audio_labels.append("Original input audio")
        audio_paths.append(str(output_audio))
        audio_labels.append("Edited output audio")
        for index, path in enumerate(reference_audios[:2]):
            audio_paths.append(path)
            audio_labels.append(f"Reference audio {index + 1}")

        choices_text = "\n".join(
            f"{chr(ord('A') + index)}. {choice}"
            for index, choice in enumerate(rubric.choices)
        )
        data = self.client.call_json(
            system_prompt=self.answer_system_prompt,
            user_text=(
                f"Rubric id: {rubric.rubric_id}\n"
                f"Category: {rubric.category}\n"
                f"Question:\n{rubric.question}\n\n"
                f"Choices:\n{choices_text}\n\n"
                "Return JSON with keys: choice, reason, issue, optimization_hint, fixable. "
                "Use choice A only when the edited output clearly passes this rubric."
            ),
            audio_paths=audio_paths,
            audio_labels=audio_labels,
        )
        choice = _normalize_choice(data.get("choice"))
        passed = choice == rubric.pass_choice
        return {
            "rubric_id": rubric.rubric_id,
            "category": rubric.category,
            "question": rubric.question,
            "choice": choice,
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "weight": rubric.weight,
            "required": rubric.required,
            "reason": str(data.get("reason") or ""),
            "issue": str(data.get("issue") or ""),
            "optimization_hint": str(data.get("optimization_hint") or ""),
            "fixable": _as_bool(data.get("fixable", True)),
        }


class Qwen3OmniEvaluator(AudioJudgeEvaluator):
    """Qwen3-Omni implementation of the generic audio judge evaluator."""

    def __init__(
        self,
        *,
        client: Qwen3OmniJudgeClient | Any | None = None,
        passing_score: float | None = None,
        max_rubrics: int | None = None,
        tool_evidence_provider: EvaluatorToolEvidenceProvider | Any | None = None,
        tool_executor: Any | None = None,
        audio_executor: Any | None = None,
        gemini_tool_planner: GeminiEvaluatorToolCallPlanner | Any | None = None,
        enable_tools: bool | None = None,
        enable_gemini_tools: bool | None = None,
    ) -> None:
        super().__init__(
            client=client or Qwen3OmniJudgeClient(),
            passing_score=passing_score,
            max_rubrics=max_rubrics,
            judge_name="qwen3_omni",
            display_name="Qwen3-Omni",
            rubric_system_prompt=_RUBRIC_SYSTEM_PROMPT,
            answer_system_prompt=_ANSWER_SYSTEM_PROMPT,
            tool_evidence_provider=tool_evidence_provider,
            tool_executor=tool_executor,
            audio_executor=audio_executor,
            gemini_tool_planner=gemini_tool_planner,
            enable_tools=enable_tools,
            enable_gemini_tools=enable_gemini_tools,
        )


def open_audio_evaluator_prompts(
    provider_display_name: str,
) -> tuple[str, str]:
    """Return the existing evaluator prompts with only model identity changed."""

    display_name = provider_display_name.strip()
    if not display_name:
        raise ValueError("provider_display_name must not be blank")
    return (
        _RUBRIC_SYSTEM_PROMPT.replace("Qwen3-Omni", display_name),
        _ANSWER_SYSTEM_PROMPT.replace("Qwen3-Omni", display_name),
    )


class GeminiAudioEvaluator(AudioJudgeEvaluator):
    """Gemini 2.5 Pro implementation of the same audio judge evaluator."""

    def __init__(
        self,
        *,
        client: GeminiAudioJudgeClient | Any | None = None,
        passing_score: float | None = None,
        max_rubrics: int | None = None,
        tool_evidence_provider: EvaluatorToolEvidenceProvider | Any | None = None,
        tool_executor: Any | None = None,
        audio_executor: Any | None = None,
        gemini_tool_planner: GeminiEvaluatorToolCallPlanner | Any | None = None,
        enable_tools: bool | None = None,
        enable_gemini_tools: bool | None = None,
    ) -> None:
        super().__init__(
            client=client or GeminiAudioJudgeClient(),
            passing_score=passing_score,
            max_rubrics=max_rubrics,
            judge_name="gemini_audio",
            display_name="Gemini 2.5 Pro",
            rubric_system_prompt=_GEMINI_RUBRIC_SYSTEM_PROMPT,
            answer_system_prompt=_GEMINI_ANSWER_SYSTEM_PROMPT,
            tool_evidence_provider=tool_evidence_provider,
            tool_executor=tool_executor,
            audio_executor=audio_executor,
            gemini_tool_planner=gemini_tool_planner,
            enable_tools=enable_tools,
            enable_gemini_tools=enable_gemini_tools,
        )


def build_default_evaluator(context: dict[str, Any] | None = None) -> Evaluator:
    """Select the default evaluator without forcing online judge dependencies."""

    context = context or {}
    provider = str(
        context.get("audio_evaluator")
        or context.get("evaluator_provider")
        or os.getenv("X_EDITOR_AUDIO_EVALUATOR", "")
    ).strip().lower()
    if provider in {"qwen", "qwen3", "qwen3_omni", "qwen3omni"}:
        return Qwen3OmniEvaluator()
    if provider in {"gemini", "gemini_audio", "gemini_2_5_pro", "gemini-2.5-pro"}:
        return GeminiAudioEvaluator()
    if provider in {"stub", "none", "off", "disabled"}:
        return StubEvaluator()
    if _context_bool(context, "use_qwen_omni_evaluator", env_name="X_EDITOR_ENABLE_QWEN_OMNI_EVALUATOR"):
        return Qwen3OmniEvaluator()
    if _context_bool(context, "use_gemini_evaluator", env_name="X_EDITOR_ENABLE_GEMINI_EVALUATOR"):
        return GeminiAudioEvaluator()
    return StubEvaluator()


def _status_evaluation(plan: AudioEditPlan) -> EvaluationResult:
    failed_steps = [step.id for step in plan.steps if step.required and step.status == "failed"]
    pending_steps = [step.id for step in plan.steps if step.required and step.status in {"pending", "running"}]
    skipped_steps = [step.id for step in plan.steps if step.required and step.status == "skipped"]
    blocking = failed_steps + pending_steps + skipped_steps
    passed = not blocking
    if passed:
        feedback = "All required audio edit steps completed successfully."
    else:
        feedback = "Some required audio edit steps did not complete: " + ", ".join(blocking)
    return EvaluationResult(
        passed=passed,
        score=None,
        feedback=feedback,
        need_replan=False,
        failed_steps=blocking,
        metadata={
            "failed_steps": failed_steps,
            "pending_steps": pending_steps,
            "skipped_steps": skipped_steps,
        },
    )


def _final_audio_path(plan: AudioEditPlan) -> str | None:
    value = plan.metadata.get("final_output_path")
    if isinstance(value, str) and value:
        return value
    for step in reversed(plan.steps):
        if isinstance(step.result, dict):
            for key in ("output_audio_path", "target_audio_path", "output_path", "generated_audio_path"):
                result_value = step.result.get(key)
                if isinstance(result_value, str) and result_value:
                    return result_value
    return plan.output_audio


def _existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def _qwen_base_urls_from_env() -> list[str]:
    for name in (
        "X_EDITOR_QWEN_OMNI_EVAL_BASE_URLS",
        "X_EDITOR_QWEN3_OMNI_BASE_URLS",
        "QWEN3_OMNI_BASE_URLS",
        "QWEN3_OMNI_BASE_URL",
    ):
        value = os.getenv(name)
        if value:
            return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _qwen_default_base_urls_when_bootstrap_configured() -> list[str]:
    if os.getenv("X_EDITOR_QWEN_OMNI_EVAL_COMMAND"):
        return ["http://127.0.0.1:8001/v1"]
    return []


def _qwen_evaluator_service_specs(context: dict[str, Any], base_urls: list[str]) -> list[dict[str, Any]]:
    raw = (
        context.get("qwen3_omni_evaluator_services")
        or context.get("qwen3_omni_managed_services")
        or context.get("evaluator_managed_services")
    )
    specs = _coerce_dict_list(raw)
    if specs:
        return specs
    command_env = "X_EDITOR_QWEN_OMNI_EVAL_COMMAND"
    if not os.getenv(command_env):
        return []
    root_url = _qwen_service_root(base_urls[0] if base_urls else "http://127.0.0.1:8001/v1")
    return [
        {
            "name": "qwen3_omni_evaluator",
            "default_url": root_url,
            "command_env": command_env,
            "health_path": "/v1/models",
            "smoke": [{"method": "GET", "path": "/v1/models"}],
            "startup_timeout_sec": float(os.getenv("X_EDITOR_QWEN_OMNI_EVAL_STARTUP_TIMEOUT", "300")),
            "setup_hint": (
                "Set X_EDITOR_QWEN_OMNI_EVAL_COMMAND to a vLLM/OpenAI-compatible "
                "Qwen3-Omni launch command, or start the service manually and set "
                "X_EDITOR_QWEN_OMNI_EVAL_BASE_URLS=http://host:port/v1."
            ),
        }
    ]


def _qwen_service_root(base_url: str) -> str:
    text = str(base_url).rstrip("/")
    if text.endswith("/v1"):
        return text[:-3]
    return text


def _base_urls_from_service_specs(specs: list[dict[str, Any]]) -> list[str]:
    urls = []
    for spec in specs:
        value = spec.get("url") or spec.get("default_url")
        if value:
            urls.append(f"{str(value).rstrip('/')}/v1")
    return urls


def _ensure_judge_service(client: Any, context: dict[str, Any]) -> dict[str, Any] | None:
    ensure = getattr(client, "ensure_service", None)
    if not callable(ensure):
        return None
    return ensure(context)


def _bootstrap_error_message_from_report(report: dict[str, Any]) -> str:
    parts = []
    for service in report.get("services") or []:
        if service.get("ok"):
            continue
        parts.append(
            f"{service.get('name')}: {service.get('message') or service.get('status') or 'service unavailable'}"
        )
    return "; ".join(parts) or "judge service bootstrap failed"


def _qwen_api_key_from_env() -> str | None:
    for name in ("X_EDITOR_QWEN_OMNI_API_KEY", "QWEN3_OMNI_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    return None


def _format_model_base_url(base_url: str, model: str) -> str:
    if "{model}" not in base_url:
        return base_url
    return base_url.format(model=model)


def _audio_data_uri(path: str) -> str:
    file_path = Path(path)
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "audio/wav"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _rubric_evidence_context(plan: AudioEditPlan, context: dict[str, Any]) -> dict[str, Any]:
    """Build benchmark-free evidence for rubric generation.

    The evaluator may reuse perception/tool evidence produced by the X-Editor
    chain, but it must not consume benchmark rubric questions, right choices, or
    benchmark-provided time windows.
    """

    perception_reports = _collect_perception_reports(plan, context)
    source_text = context.get("source_text") or plan.global_params.get("source_text")
    trusted_time_spans: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    speaker_segments: list[dict[str, Any]] = []
    described_segments: list[dict[str, Any]] = []
    perception_sources: list[dict[str, Any]] = []
    for source, perception in perception_reports:
        if not source_text:
            source_text = perception.get("source_text") or perception.get("text")
        report_words = _compact_words(
            perception.get("word_timestamps"),
            trusted_time_spans,
            source=f"{source}.word_timestamps",
        )
        report_speakers = _compact_segments(
            perception.get("speaker_segments") or perception.get("segments"),
            trusted_time_spans,
            source=f"{source}.speaker_segments",
        )
        report_descriptions = _compact_segments(
            perception.get("described_segments") or perception.get("semantic_segments"),
            trusted_time_spans,
            source=f"{source}.described_segments",
        )
        words.extend(report_words)
        speaker_segments.extend(report_speakers)
        described_segments.extend(report_descriptions)
        perception_sources.append(
            _without_empty(
                {
                    "source": source,
                    "source_text": _truncate_text(str(perception.get("source_text") or perception.get("text") or ""), 360),
                    "word_timestamp_count": len(report_words),
                    "speaker_segment_count": len(report_speakers),
                    "described_segment_count": len(report_descriptions),
                    "services": _safe_prompt_value(perception.get("services"), depth=2),
                }
            )
        )
    tool_evidence = _tool_evidence_from_plan(plan, trusted_time_spans)
    provider_evidence, provider_errors = _evaluation_evidence_from_providers(plan, context)
    external_evidence = []
    for raw_evidence in [context.get("evaluation_evidence"), *provider_evidence]:
        safe_evidence = _external_evaluation_evidence(raw_evidence, trusted_time_spans)
        if safe_evidence:
            external_evidence.append(safe_evidence)
    evidence = {
        "policy": {
            "benchmark_oracle_forbidden": True,
            "forbidden_inputs": [
                "benchmark rubrics",
                "benchmark right_choice/wrong_choices",
                "benchmark-provided time windows",
                "MMAE metadata questions",
            ],
            "time_window_rule": (
                "Use explicit time windows only when they appear in trusted_time_spans below; "
                "otherwise write audio-grounded questions without timestamps."
            ),
        },
        "source_text": _truncate_text(str(source_text), 2000) if source_text else None,
        "perception_sources": perception_sources,
        "word_timestamps": words[:120],
        "speaker_segments": speaker_segments[:80],
        "described_segments": described_segments[:80],
        "tool_evidence": tool_evidence,
        "external_evaluation_evidence": external_evidence,
        "evaluation_evidence_provider_errors": provider_errors,
        "trusted_time_spans": trusted_time_spans[:80],
    }
    return _without_empty(evidence)


def _collect_perception_reports(plan: AudioEditPlan, context: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    reports: list[tuple[str, dict[str, Any]]] = []
    _append_perception_report(reports, "context.perception_report", context.get("perception_report"))
    _append_perception_report(reports, "plan.global_params.perception_report", plan.global_params.get("perception_report"))
    _append_perception_report(reports, "plan.metadata.perception_report", plan.metadata.get("perception_report"))
    for step in plan.steps:
        if not isinstance(step.result, dict):
            continue
        source = f"step_result.{step.id}"
        _append_perception_report(reports, f"{source}.perception_report", step.result.get("perception_report"))
        if step.operation.startswith("perception.") and not isinstance(step.result.get("perception_report"), dict):
            _append_perception_report(reports, source, step.result)
    return reports


def _append_perception_report(reports: list[tuple[str, dict[str, Any]]], source: str, value: Any) -> None:
    if not isinstance(value, dict):
        return
    if not any(
        value.get(key)
        for key in (
            "source_text",
            "text",
            "word_timestamps",
            "speaker_segments",
            "segments",
            "described_segments",
            "semantic_segments",
        )
    ):
        return
    reports.append((source, value))


def _rubric_generation_prompt(
    plan: AudioEditPlan,
    context: dict[str, Any],
    has_input_audio: bool,
    has_output_audio: bool,
    rubric_evidence: dict[str, Any],
) -> str:
    compact_steps = [
        {
            "id": step.id,
            "operation": step.operation,
            "description": step.description,
            "status": step.status,
            "params": _small_mapping(step.params),
        }
        for step in plan.steps
    ]
    public_context = _safe_prompt_mapping({
        "reference_audio_paths": context.get("reference_audio_paths") or [],
        "evaluation_focus": context.get("evaluation_focus"),
    })
    public_context = _without_empty(public_context)
    return (
        "Create general-purpose verification rubrics for an X-Editor audio edit.\n"
        f"User request: {plan.request}\n"
        f"Has original input audio: {has_input_audio}\n"
        f"Has edited output audio: {has_output_audio}\n"
        f"Executed steps JSON: {json.dumps(compact_steps, ensure_ascii=False)}\n"
        f"Context JSON: {json.dumps(public_context, ensure_ascii=False)[:6000]}\n\n"
        f"Trusted Evidence JSON: {json.dumps(rubric_evidence, ensure_ascii=False)[:12000]}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "passing_score": 0.75,\n'
        '  "rubrics": [\n'
        "    {\n"
        '      "rubric_id": "R1",\n'
        '      "category": "instruction_following|preservation|audio_quality",\n'
        '      "question": "A yes/no question answerable by listening to the provided audio.",\n'
        '      "weight": 1.0,\n'
        '      "required": true,\n'
        '      "evidence_source": "instruction_only|perception.word_timestamps|perception.speaker_segments|tool_result|external_tool_evidence",\n'
        '      "rationale": "why this verifies the edit"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules: create 3-5 rubrics; do not mention MMAE or any benchmark; keep questions "
        "answerable from audio only; include one instruction_following rubric, one preservation "
        "rubric when original input audio exists, and one audio_quality rubric; do not require "
        "external facts or hidden references. Never copy benchmark-provided rubrics, answer choices, "
        "or time windows. If a question uses a time window such as <audio output[1.0s:2.0s]>, "
        "that exact window must come from Trusted Evidence JSON trusted_time_spans; otherwise ask "
        "the question without timestamps."
    )


def _small_mapping(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"perception_report", "word_timestamps", "speaker_segments"} or _forbidden_oracle_key(str(key)):
            continue
        safe_value = _safe_prompt_value(item, depth=2)
        if safe_value not in (None, "", [], {}):
            result[key] = safe_value
    return result


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _run_coroutine_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def run_in_thread() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive bridge
            result["error"] = exc

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _coerce_dict_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _evaluator_tooling_requested(context: dict[str, Any], default: bool | None) -> bool:
    if context.get("evaluation_tool_calls") or context.get("evaluation_runtime_steps"):
        return True
    if _context_bool(context, "enable_evaluator_tools", env_name="X_EDITOR_ENABLE_EVALUATOR_TOOLS"):
        return True
    if _enable_gemini_evaluator_tools(context, None):
        return True
    return bool(default)


def _enable_gemini_evaluator_tools(context: dict[str, Any], default: bool | None) -> bool:
    if "enable_gemini_evaluator_tools" in context:
        return _as_bool(context.get("enable_gemini_evaluator_tools"))
    env_value = os.getenv("X_EDITOR_ENABLE_GEMINI_EVALUATOR_TOOLS")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def _allowed_evaluator_tool_ids(context: dict[str, Any], configured: set[str] | None) -> set[str]:
    values = configured or set(DEFAULT_EVALUATOR_TOOL_IDS)
    raw = context.get("evaluation_allowed_tool_ids") or os.getenv("X_EDITOR_EVALUATOR_ALLOWED_TOOL_IDS")
    if raw:
        if isinstance(raw, str):
            values = {item.strip() for item in raw.split(",") if item.strip()}
        elif isinstance(raw, list):
            values = {str(item).strip() for item in raw if str(item).strip()}
    return set(values)


def _allowed_evaluator_runtime_operations(context: dict[str, Any], configured: set[str] | None) -> set[str]:
    values = configured or set(DEFAULT_EVALUATOR_RUNTIME_OPERATIONS)
    raw = context.get("evaluation_allowed_runtime_operations") or os.getenv(
        "X_EDITOR_EVALUATOR_ALLOWED_RUNTIME_OPERATIONS"
    )
    if raw:
        if isinstance(raw, str):
            values = {item.strip() for item in raw.split(",") if item.strip()}
        elif isinstance(raw, list):
            values = {str(item).strip() for item in raw if str(item).strip()}
    return set(values)


def _prepare_evaluator_tool_arguments(
    *,
    tool_id: str,
    raw_arguments: dict[str, Any],
    plan: AudioEditPlan,
    context: dict[str, Any],
    output_dir: Path,
    index: int,
) -> dict[str, Any]:
    arguments = _resolve_evaluator_placeholders(
        raw_arguments,
        plan=plan,
        context=context,
        output_dir=output_dir,
    )
    try:
        from x_editor.tools.loader import load_manual

        manual = load_manual(tool_id)
        schema = manual.input_schema or {}
        properties = schema.get("properties") if isinstance(schema, dict) else {}
        required = schema.get("required") if isinstance(schema, dict) else []
    except Exception:
        properties = {}
        required = []

    input_audio = context.get("input_audio") or plan.input_audio
    output_audio = _final_audio_path(plan)
    if (
        ("input_audio_path" in properties or "input_audio_path" in required)
        and not arguments.get("input_audio_path")
    ):
        arguments["input_audio_path"] = output_audio or input_audio
    if ("output_dir" in properties or "output_dir" in required) and not arguments.get("output_dir"):
        arguments["output_dir"] = str(output_dir / f"{index:02d}_{tool_id}")
    if ("output_path" in properties or "output_path" in required) and not arguments.get("output_path"):
        arguments["output_path"] = str(output_dir / f"{index:02d}_{tool_id}.wav")

    protected_paths = {str(path) for path in [input_audio, output_audio] if path}
    if arguments.get("output_path") in protected_paths:
        arguments["output_path"] = str(output_dir / f"{index:02d}_{tool_id}.wav")
    return arguments


def _resolve_evaluator_placeholders(
    value: Any,
    *,
    plan: AudioEditPlan,
    context: dict[str, Any],
    output_dir: Path,
) -> Any:
    replacements = {
        "input_audio": str(context.get("input_audio") or plan.input_audio or ""),
        "output_audio": str(_final_audio_path(plan) or ""),
        "output_dir": str(output_dir),
        "source_text": str(context.get("source_text") or plan.global_params.get("source_text") or ""),
    }
    if isinstance(value, str):
        result = value
        for key, replacement in replacements.items():
            result = result.replace("{" + key + "}", replacement)
        return result
    if isinstance(value, dict):
        return {
            key: _resolve_evaluator_placeholders(item, plan=plan, context=context, output_dir=output_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_evaluator_placeholders(item, plan=plan, context=context, output_dir=output_dir)
            for item in value
        ]
    return value


def _safe_prompt_mapping(value: dict[str, Any], *, limit: int = 12) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, item in list(value.items())[:limit]:
        if _forbidden_oracle_key(str(key)):
            continue
        safe_value = _safe_prompt_value(item, depth=2)
        if safe_value not in (None, "", [], {}):
            safe[str(key)] = safe_value
    return safe


def _safe_prompt_value(value: Any, *, depth: int) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if _looks_like_benchmark_oracle(value):
            return "[redacted benchmark/oracle text]"
        return _truncate_text(value, 800)
    if depth <= 0:
        return _truncate_text(str(value), 240)
    if isinstance(value, dict):
        return _safe_prompt_mapping(value, limit=8)
    if isinstance(value, list):
        safe_items = [_safe_prompt_value(item, depth=depth - 1) for item in value[:10]]
        return [item for item in safe_items if item not in (None, "", [], {})]
    return _truncate_text(str(value), 240)


def _evaluation_evidence_from_providers(
    plan: AudioEditPlan,
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    providers = []
    single_provider = context.get("evaluation_evidence_provider")
    if single_provider is not None:
        providers.append(single_provider)
    many_providers = context.get("evaluation_evidence_providers") or []
    if isinstance(many_providers, (list, tuple)):
        providers.extend(many_providers)
    else:
        providers.append(many_providers)

    evidence: list[dict[str, Any]] = []
    errors: list[str] = []
    for provider in providers:
        try:
            if callable(provider):
                try:
                    raw = provider(plan=plan, context=context)
                except TypeError:
                    raw = provider(plan, context)
            elif hasattr(provider, "collect_evaluation_evidence"):
                try:
                    raw = provider.collect_evaluation_evidence(plan=plan, context=context)
                except TypeError:
                    raw = provider.collect_evaluation_evidence(plan, context)
            else:
                raw = provider
            if hasattr(raw, "__await__"):
                errors.append(f"{type(provider).__name__}: async evidence providers are not supported here")
                continue
            if isinstance(raw, dict):
                evidence.append(raw)
            elif isinstance(raw, list):
                evidence.extend(item for item in raw if isinstance(item, dict))
        except Exception as exc:
            errors.append(f"{type(provider).__name__}: {type(exc).__name__}: {exc}")
    return evidence, errors


def _compact_words(
    value: Any,
    trusted_time_spans: list[dict[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    words: list[dict[str, Any]] = []
    for index, item in enumerate(value[:80]):
        if not isinstance(item, dict):
            continue
        start, end = _span_bounds(item)
        text = str(item.get("text") or item.get("word") or item.get("token") or "").strip()
        compact = _without_empty(
            {
                "index": index,
                "text": text,
                "start_sec": start,
                "end_sec": end,
                "speaker": item.get("speaker") or item.get("speaker_id"),
            }
        )
        if compact:
            words.append(compact)
        if start is not None and end is not None:
            trusted_time_spans.append(
                _without_empty(
                    {
                        "source": source,
                        "start_sec": start,
                        "end_sec": end,
                        "text": text,
                    }
                )
            )
    return words


def _compact_segments(
    value: Any,
    trusted_time_spans: list[dict[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    segments: list[dict[str, Any]] = []
    for item in value[:40]:
        if not isinstance(item, dict):
            continue
        start, end = _span_bounds(item)
        compact = _without_empty(
            {
                "start_sec": start,
                "end_sec": end,
                "speaker": item.get("speaker") or item.get("speaker_id") or item.get("label"),
                "text": _truncate_text(str(item.get("text") or item.get("transcript") or ""), 240),
                "description": _truncate_text(str(item.get("description") or item.get("summary") or ""), 240),
                "source": item.get("source") or source,
            }
        )
        if compact:
            segments.append(compact)
        if start is not None and end is not None:
            trusted_time_spans.append(
                _without_empty(
                    {
                        "source": source,
                        "start_sec": start,
                        "end_sec": end,
                        "speaker": compact.get("speaker"),
                        "text": compact.get("text"),
                        "description": compact.get("description"),
                    }
                )
            )
    return segments


def _tool_evidence_from_plan(plan: AudioEditPlan, trusted_time_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for step in plan.steps:
        if not isinstance(step.result, dict):
            continue
        result = step.result
        spans = _candidate_spans_from_mapping(result)
        for span in spans:
            start, end = _span_bounds(span)
            if start is not None and end is not None:
                trusted_time_spans.append(
                    _without_empty(
                        {
                            "source": f"tool_result.{step.operation}",
                            "step_id": step.id,
                            "start_sec": start,
                            "end_sec": end,
                            "label": span.get("label") or span.get("role") or span.get("source"),
                            "text": _truncate_text(str(span.get("text") or ""), 240),
                        }
                    )
                )
        evidence.append(
            _without_empty(
                {
                    "step_id": step.id,
                    "operation": step.operation,
                    "status": step.status,
                    "selected_span_count": len(spans),
                    "output_audio_path": result.get("output_audio_path")
                    or result.get("target_audio_path")
                    or result.get("output_path"),
                    "notes": _truncate_text(str(result.get("notes") or result.get("warning") or ""), 360),
                }
            )
        )
    return evidence[:40]


def _external_evaluation_evidence(value: Any, trusted_time_spans: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    trusted_external_spans: list[dict[str, Any]] = []
    for span in _candidate_spans_from_mapping(value):
        source = str(span.get("source") or "")
        if not _trusted_external_time_span_source(source):
            continue
        start, end = _span_bounds(span)
        if start is not None and end is not None:
            trusted_external_spans.append(span)
            trusted_time_spans.append(
                _without_empty(
                    {
                        "source": source,
                        "start_sec": start,
                        "end_sec": end,
                        "label": span.get("label") or span.get("role"),
                        "text": _truncate_text(str(span.get("text") or ""), 240),
                    }
                )
            )
    safe: dict[str, Any] = {}
    for key, item in value.items():
        if _forbidden_oracle_key(str(key)):
            continue
        if key in {"trusted_time_spans", "spans", "selected_spans", "segments"}:
            safe[key] = _compact_segments(
                trusted_external_spans,
                [],
                source="external_tool_evidence",
            )
        elif isinstance(item, (str, int, float, bool)) or item is None:
            safe[key] = _safe_prompt_value(item, depth=1)
        elif isinstance(item, list):
            safe[key] = _safe_prompt_value(item, depth=2)
        elif isinstance(item, dict):
            safe[key] = _safe_prompt_mapping(item)
    return _without_empty(safe)


def _candidate_spans_from_mapping(value: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for key in (
        "trusted_time_spans",
        "selected_spans",
        "target_spans",
        "spans",
        "segments",
        "edited_spans",
        "edits",
    ):
        raw = value.get(key)
        if isinstance(raw, list):
            spans.extend(item for item in raw if isinstance(item, dict))
    return spans


def _trusted_spans_from_tool_output(output: dict[str, Any], *, source_prefix: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for item in _candidate_spans_from_mapping(output):
        start, end = _span_bounds(item)
        if start is None or end is None:
            continue
        source = str(item.get("source") or source_prefix)
        if not _trusted_external_time_span_source(source):
            source = source_prefix
        spans.append(
            _without_empty(
                {
                    "source": source,
                    "start_sec": start,
                    "end_sec": end,
                    "label": item.get("label") or item.get("role"),
                    "text": _truncate_text(str(item.get("text") or ""), 240),
                }
            )
        )

    perception = output.get("perception_report") if isinstance(output.get("perception_report"), dict) else output
    if isinstance(perception, dict):
        for word in perception.get("word_timestamps") or []:
            if not isinstance(word, dict):
                continue
            start, end = _span_bounds(word)
            if start is None or end is None:
                continue
            spans.append(
                _without_empty(
                    {
                        "source": f"{source_prefix}.perception.word_timestamps",
                        "start_sec": start,
                        "end_sec": end,
                        "text": str(word.get("text") or word.get("word") or ""),
                    }
                )
            )
        for segment in perception.get("speaker_segments") or perception.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            start, end = _span_bounds(segment)
            if start is None or end is None:
                continue
            spans.append(
                _without_empty(
                    {
                        "source": f"{source_prefix}.perception.speaker_segments",
                        "start_sec": start,
                        "end_sec": end,
                        "speaker": segment.get("speaker") or segment.get("speaker_id"),
                        "text": _truncate_text(str(segment.get("text") or segment.get("transcript") or ""), 240),
                        "description": _truncate_text(str(segment.get("description") or ""), 240),
                    }
                )
            )

    edit_result = output.get("edit_result") if isinstance(output.get("edit_result"), dict) else {}
    for edit in edit_result.get("edits_applied") or []:
        if not isinstance(edit, dict):
            continue
        start, end = _span_bounds(edit)
        if start is None or end is None:
            source_span = edit.get("source_span")
            if isinstance(source_span, list) and len(source_span) >= 2:
                start, end = _float_or_none(source_span[0]), _float_or_none(source_span[1])
        if start is None or end is None:
            continue
        spans.append(
            _without_empty(
                {
                    "source": f"{source_prefix}.edit_result",
                    "start_sec": start,
                    "end_sec": end,
                    "label": edit.get("edit_type"),
                    "text": _truncate_text(str(edit.get("words") or ""), 240),
                }
            )
        )
    alignment = edit_result.get("alignment") if isinstance(edit_result.get("alignment"), dict) else {}
    for region in alignment.get("edit_regions") or []:
        if not isinstance(region, dict):
            continue
        start, end = _span_bounds(region)
        if start is None or end is None:
            continue
        spans.append(
            _without_empty(
                {
                    "source": f"{source_prefix}.alignment",
                    "start_sec": start,
                    "end_sec": end,
                    "label": region.get("tag_type"),
                    "text": _truncate_text(str(region.get("text") or ""), 240),
                }
            )
        )
    return spans[:120]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_TRUSTED_EXTERNAL_TIME_SPAN_PREFIXES = (
    "aligner.",
    "alignment.",
    "artifact_ops.",
    "asr.",
    "diarization.",
    "edit_data_pipeline.",
    "external_tool.",
    "perception.",
    "speech_perception.",
    "step_result.",
    "tool_result.",
    "vad.",
)

_TRUSTED_EXTERNAL_TIME_SPAN_NAMES = {
    "aligner",
    "alignment",
    "asr",
    "diarization",
    "speech_perception",
    "vad",
}

_FORBIDDEN_TIME_SPAN_SOURCE_MARKERS = (
    "answer_key",
    "benchmark",
    "ground_truth",
    "mmae",
    "oracle",
    "right_choice",
    "rubric",
    "wrong_choice",
)


def _trusted_external_time_span_source(source: str) -> bool:
    normalized = source.strip().lower()
    if not normalized:
        return False
    if any(marker in normalized for marker in _FORBIDDEN_TIME_SPAN_SOURCE_MARKERS):
        return False
    return normalized in _TRUSTED_EXTERNAL_TIME_SPAN_NAMES or normalized.startswith(
        _TRUSTED_EXTERNAL_TIME_SPAN_PREFIXES
    )


def _span_bounds(item: dict[str, Any]) -> tuple[float | None, float | None]:
    start = _first_float(item, "start_sec", "start", "start_time", "begin_sec", "begin")
    end = _first_float(item, "end_sec", "end", "end_time", "stop_sec", "stop")
    start_ms = _first_float(item, "start_ms", "begin_ms")
    end_ms = _first_float(item, "end_ms", "stop_ms")
    if start is None and start_ms is not None:
        start = start_ms / 1000.0
    if end is None and end_ms is not None:
        end = end_ms / 1000.0
    return start, end


def _first_float(item: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in item:
            continue
        value = item.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _without_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item not in (None, "", [], {})
    }


def _truncate_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _forbidden_oracle_key(key: str) -> bool:
    normalized = key.lower()
    return any(
        marker in normalized
        for marker in (
            "rubric",
            "right_choice",
            "wrong_choice",
            "benchmark",
            "mmae",
            "answer_key",
            "ground_truth",
        )
    )


def _rubrics_from_model(
    data: dict[str, Any],
    *,
    max_rubrics: int,
    trusted_time_spans: list[dict[str, Any]] | None = None,
) -> list[VerificationRubric]:
    raw_rubrics = data.get("rubrics")
    if not isinstance(raw_rubrics, list):
        return []
    rubrics: list[VerificationRubric] = []
    for index, item in enumerate(raw_rubrics, start=1):
        if len(rubrics) >= max_rubrics:
            break
        if not isinstance(item, dict):
            continue
        if any(_forbidden_model_rubric_key(str(key)) for key in item.keys()):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        if _looks_like_benchmark_oracle(question):
            continue
        if not _time_references_are_trusted(question, trusted_time_spans or []):
            continue
        evidence_source = str(item.get("evidence_source") or "instruction_only")
        if _forbidden_oracle_key(evidence_source) or _looks_like_benchmark_oracle(evidence_source):
            continue
        rubric_id = str(item.get("rubric_id") or item.get("id") or f"R{index}")
        rubrics.append(
            VerificationRubric(
                rubric_id=rubric_id,
                category=str(item.get("category") or "instruction_following"),
                question=question,
                weight=max(0.1, float(item.get("weight") or 1.0)),
                required=bool(item.get("required", True)),
                evidence_source=evidence_source,
                rationale=str(item.get("rationale") or ""),
            )
        )
    return rubrics


_TIME_WINDOW_PATTERN = re.compile(
    r"\[\s*(?P<start>\d+(?:\.\d+)?)\s*s?\s*:\s*(?P<end>\d+(?:\.\d+)?)?\s*s?\s*\]"
)


def _time_references_are_trusted(question: str, trusted_time_spans: list[dict[str, Any]]) -> bool:
    refs = [
        (float(match.group("start")), float(match.group("end")) if match.group("end") else None)
        for match in _TIME_WINDOW_PATTERN.finditer(question)
    ]
    if not refs:
        return True
    spans: list[tuple[float, float]] = []
    for item in trusted_time_spans:
        if not isinstance(item, dict):
            continue
        start, end = _span_bounds(item)
        if start is not None and end is not None:
            spans.append((start, end))
    if not spans:
        return False
    for start, end in refs:
        if end is None:
            return False
        if not any(abs(start - span_start) <= 0.05 and abs(end - span_end) <= 0.05 for span_start, span_end in spans):
            return False
    return True


def _looks_like_benchmark_oracle(question: str) -> bool:
    lowered = question.lower()
    return "mmae" in lowered or "benchmark" in lowered or "right_choice" in lowered or "wrong_choices" in lowered


def _forbidden_model_rubric_key(key: str) -> bool:
    normalized = key.lower()
    return normalized in {
        "right_choice",
        "wrong_choices",
        "answer_key",
        "ground_truth",
        "benchmark_question",
    } or normalized.startswith("mmae_")


def _fallback_rubrics(request: str, *, has_input_audio: bool, max_rubrics: int) -> list[VerificationRubric]:
    rubrics = [
        VerificationRubric(
            rubric_id="R1",
            category="instruction_following",
            question=(
                "Does the edited output clearly satisfy the user's requested audio edit: "
                f"{request}?"
            ),
            required=True,
            rationale="Primary task success.",
        ),
        VerificationRubric(
            rubric_id="R2",
            category="audio_quality",
            question="Is the edited output free of severe glitches, dropouts, clipping, or unnatural discontinuities?",
            required=True,
            rationale="The output should remain usable audio.",
        ),
    ]
    if has_input_audio:
        rubrics.insert(
            1,
            VerificationRubric(
                rubric_id="R_preserve",
                category="preservation",
                question=(
                    "Outside the requested edit, does the edited output preserve the original "
                    "speech content, speaker identity, timing, and background as much as the task allows?"
                ),
                required=True,
                rationale="Local edits should not damage unrelated content.",
            ),
        )
    return rubrics[:max_rubrics]


def _gemini_evaluator_tool_prompt(
    *,
    plan: AudioEditPlan,
    context: dict[str, Any],
    model_name: str,
    allowed_tool_ids: set[str],
    allowed_runtime_operations: set[str],
) -> str:
    tool_cards = _evaluator_tool_cards(allowed_tool_ids)
    public_context = _safe_prompt_mapping(
        {
            "input_audio": context.get("input_audio") or plan.input_audio,
            "output_audio": _final_audio_path(plan),
            "source_text": context.get("source_text") or plan.global_params.get("source_text"),
            "reference_audio_paths": context.get("reference_audio_paths") or plan.global_params.get("reference_audio_paths"),
            "perception_report": _safe_prompt_value(context.get("perception_report") or plan.global_params.get("perception_report"), depth=3),
            "evaluation_focus": context.get("evaluation_focus"),
        },
        limit=8,
    )
    return (
        f"## Evaluator Tool Planner Model\n{model_name}\n\n"
        f"## User Request\n{plan.request}\n\n"
        f"## Executed Plan JSON\n{json.dumps(plan.model_dump(), ensure_ascii=False)[:10000]}\n\n"
        f"## Public Context JSON\n{json.dumps(public_context, ensure_ascii=False)[:8000]}\n\n"
        f"## Allowed Catalog Tool IDs\n{json.dumps(sorted(allowed_tool_ids), ensure_ascii=False)}\n\n"
        f"## Allowed Runtime Operations\n{json.dumps(sorted(allowed_runtime_operations), ensure_ascii=False)}\n\n"
        f"## Tool Cards JSON\n{json.dumps(tool_cards, ensure_ascii=False)[:8000]}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "tool_calls": [{"tool_id": "speech_separation", "arguments": {}}],\n'
        '  "runtime_steps": [{"id": "eval_perception", "operation": "perception.speech_context", "description": "...", "inputs": {"input_audio_path": "current"}, "params": {}}]\n'
        "}\n"
        "Rules: propose only evidence-collection calls needed to verify the edit. Do not modify the judged output. "
        "Do not use or request MMAE/benchmark rubrics, right choices, wrong choices, ground truth, or benchmark timestamps. "
        "Prefer runtime operation perception.speech_context when only transcript/source_text is missing. Prefer "
        "perception.audio_fallback when word timestamps, speaker segments, or semantic segment descriptions are missing. "
        "Use placeholders {input_audio}, {output_audio}, {output_dir}, {source_text} "
        "inside arguments when paths are needed. Return empty lists when no extra evidence tools are needed."
    )


def _evaluator_tool_cards(allowed_tool_ids: set[str]) -> list[dict[str, Any]]:
    try:
        from x_editor.tools.loader import load_all_capabilities
    except Exception:
        return []
    cards = []
    for tool_id, capability in load_all_capabilities().items():
        if tool_id not in allowed_tool_ids:
            continue
        cards.append(
            _safe_prompt_mapping(
                {
                    "tool_id": tool_id,
                    "purpose": capability.purpose,
                    "when_to_use": capability.when_to_use,
                    "inputs": capability.inputs,
                    "outputs": capability.outputs,
                    "constraints": capability.constraints,
                },
                limit=8,
            )
        )
    return cards


def _validated_gemini_evaluator_tool_calls(
    data: dict[str, Any],
    *,
    allowed_tool_ids: set[str],
    allowed_runtime_operations: set[str],
) -> dict[str, list[dict[str, Any]]]:
    tool_calls = []
    for item in _coerce_dict_list(data.get("tool_calls"))[:4]:
        tool_id = str(item.get("tool_id") or "").strip()
        if tool_id not in allowed_tool_ids:
            continue
        arguments = _mapping(item.get("arguments"))
        if _contains_oracle_material(arguments):
            continue
        tool_calls.append({"tool_id": tool_id, "arguments": arguments})

    runtime_steps = []
    for item in _coerce_dict_list(data.get("runtime_steps"))[:4]:
        operation = str(item.get("operation") or "").strip()
        if operation not in allowed_runtime_operations:
            continue
        if _contains_oracle_material(item):
            continue
        runtime_steps.append(item)
    return {"tool_calls": tool_calls, "runtime_steps": runtime_steps}


def _contains_oracle_material(value: Any) -> bool:
    if isinstance(value, str):
        return _looks_like_benchmark_oracle(value)
    if isinstance(value, dict):
        return any(_forbidden_oracle_key(str(key)) or _contains_oracle_material(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_oracle_material(item) for item in value)
    return False


def _weighted_score(answers: list[dict[str, Any]]) -> float:
    total = sum(float(item.get("weight") or 1.0) for item in answers)
    if total <= 0:
        return 0.0
    earned = sum(float(item.get("weight") or 1.0) * float(item.get("score") or 0.0) for item in answers)
    return earned / total


def _evaluation_feedback(
    passed: bool,
    score: float,
    passing_score: float,
    answers: list[dict[str, Any]],
    display_name: str,
) -> str:
    if passed:
        return f"{display_name} verification passed with score {score:.2f} >= {passing_score:.2f}."
    failed = [item for item in answers if not item.get("passed")]
    parts = [f"{display_name} verification failed with score {score:.2f} < {passing_score:.2f}."]
    for item in failed[:5]:
        issue = item.get("issue") or item.get("reason") or "rubric did not pass"
        hint = item.get("optimization_hint") or "No concrete automatic repair hint was provided."
        parts.append(f"{item.get('rubric_id')}: {issue} Suggested repair: {hint}")
    return " ".join(parts)


def _normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    match = re.search(r"[A-Z]", text)
    return match.group(0) if match else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _float_from_env(*names: str, default: float) -> float:
    for name in names:
        value = os.getenv(name)
        if value:
            return float(value)
    return default


def _int_from_env(*names: str, default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value:
            return int(value)
    return default


def _parse_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
        candidates.append(stripped.strip())
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _context_bool(context: dict[str, Any], key: str, *, env_name: str) -> bool:
    if key in context:
        value = context[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return os.getenv(env_name, "0").lower() in {"1", "true", "yes", "on"}


def _context_bool_optional(context: dict[str, Any], key: str, *, env_name: str) -> bool | None:
    if key in context:
        value = context[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    env_value = os.getenv(env_name)
    if env_value is None:
        return None
    return env_value.lower() in {"1", "true", "yes", "on"}


_RUBRIC_SYSTEM_PROMPT = """You are Qwen3-Omni acting as a general X-Editor audio verification designer.
Return only valid JSON. Build rubrics that can be answered by listening to original, edited, and optional reference audio.
Do not assume MMAE benchmark metadata. Never use benchmark-provided questions, answers, or timestamps.
Do not demand information that is not audible or present in the request or trusted X-Editor evidence."""


_ANSWER_SYSTEM_PROMPT = """You are Qwen3-Omni acting as an audio edit judge.
Listen carefully to the provided audio. Answer the rubric conservatively.
Return only JSON. Use choice A only when the edited output clearly passes; use B for clear failure; use C when the evidence is insufficient.
Keep reason, issue, and optimization_hint concise."""


_GEMINI_RUBRIC_SYSTEM_PROMPT = """You are Gemini 2.5 Pro acting as a general X-Editor audio verification designer.
Return only valid JSON. Build rubrics that can be answered by listening to original, edited, and optional reference audio.
Do not assume MMAE benchmark metadata. Never use benchmark-provided questions, answers, or timestamps.
Do not demand information that is not audible or present in the request or trusted X-Editor evidence."""


_GEMINI_ANSWER_SYSTEM_PROMPT = """You are Gemini 2.5 Pro acting as an audio edit judge.
Listen carefully to the provided audio. Answer the rubric conservatively.
Return only JSON. Use choice A only when the edited output clearly passes; use B for clear failure; use C when the evidence is insufficient.
Keep reason, issue, and optimization_hint concise."""


_GEMINI_EVALUATOR_TOOL_SYSTEM_PROMPT = """You are Gemini 2.5 Pro acting as the X-Editor evaluator evidence tool planner.
Return only valid JSON. You may choose evidence-only X-Editor tools or runtime perception operations when the evaluator lacks enough trusted evidence.
Never use benchmark rubrics, answer keys, ground truth, or benchmark-provided timestamps.
Do not request tools that modify the final judged output; any output path must be an evaluator scratch artifact."""
