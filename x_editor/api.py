"""One public entry point for the existing Forward and policy-revision runtimes.

No planning, tool routing, evidence validation, or scoring algorithm lives here.
Configuration is explicit and every invocation owns a new output directory.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from x_editor.core.provider_requests import ProviderSchedulingConfigV2, ProviderSchedulingRuntime
from x_editor.runtime.open_provider_workflow import (
    OpenProviderWorkflowProfile, build_open_provider_workflow_runtime,
)
from x_editor.runtime.workflow_case import WorkflowRuntimeScenario
from x_editor.tools.selection import ToolSelection, tool_selection_context


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["x-edit.runtime/v1"] = "x-edit.runtime/v1"
    profile: OpenProviderWorkflowProfile
    scheduling: ProviderSchedulingConfigV2 | None = None
    revision: dict = Field(default_factory=dict)
    tools: ToolSelection | None = None

    @model_validator(mode="after")
    def select_tools(self) -> "RuntimeConfig":
        if self.tools is not None:
            self.profile = self.tools.project_profile(self.profile)
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "RuntimeConfig":
        path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("runtime configuration must be an object")
        for field in ("profile", "scheduling", "revision", "tools"):
            value = raw.get(field)
            if isinstance(value, str):
                raw[field] = yaml.safe_load((path.parent / value).read_text())
        config = cls.model_validate(raw)
        # Endpoint relocation only. Prompt, budgets, sampling and capabilities stay fixed.
        payload = config.model_dump(mode="json")
        profile = payload["profile"]
        for name, provider in (("PLANNER", profile["planner"]),
                               ("PERCEPTION", profile["perception"]),
                               ("REASONER", profile["semantic_planning"].get("reasoner"))):
            if provider is not None:
                provider["base_url"] = os.getenv(f"XEDIT_{name}_URL", provider["base_url"])
                provider["model"] = os.getenv(f"XEDIT_{name}_MODEL", provider["model"])
        for name, field in (("SEMANTIC_JUDGE", "semantic_judge"), ("AUDIO_JUDGE", "audio_judge")):
            provider = payload["revision"].get(field)
            if provider:
                provider["base_url"] = os.getenv(f"XEDIT_{name}_URL", provider["base_url"])
                provider["model"] = os.getenv(f"XEDIT_{name}_MODEL", provider["model"])
        return cls.model_validate(payload)


def _write(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def bind_assets(scenario: WorkflowRuntimeScenario, base: Path | None = None) -> WorkflowRuntimeScenario:
    """Check readable inputs and bind their content, retaining user roles/order."""
    import soundfile as sf

    payload = scenario.model_dump(mode="json")
    for asset in payload["audio_assets"]:
        path = Path(asset["path"]).expanduser()
        path = ((base or Path.cwd()) / path).resolve()
        info = sf.info(path)
        if info.frames <= 0:
            raise ValueError(f"empty input audio: {asset['asset_ref']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if asset.get("content_sha256") not in (None, digest):
            raise ValueError(f"input hash mismatch: {asset['asset_ref']}")
        asset.update(path=str(path), content_sha256=digest)
    return WorkflowRuntimeScenario.model_validate(payload)


class XEdit:
    """The same runtime for CLI, Python, multi-audio, and multi-turn requests."""

    def __init__(self, config: RuntimeConfig):
        self.config = config

    @classmethod
    def from_config(cls, path: str | Path) -> "XEdit":
        return cls(RuntimeConfig.from_file(path))

    def _factory(self, directory: Path, *, execution_resume=None):
        from x_editor.evaluation.probe_invocation import record_runtime_modules
        from x_editor.revision.module_chain_trial import RecordedForward
        from x_editor.revision.record_benchmark import RecordedScenario

        scheduling = ProviderSchedulingRuntime(self.config.scheduling) if self.config.scheduling else None
        runtime = build_open_provider_workflow_runtime(
            self.config.profile, output_dir=directory, provider_scheduling=scheduling,
        )
        runtime = record_runtime_modules(runtime, directory / "module_invocations")
        path = runtime.forward_path
        if execution_resume is not None:
            from x_editor.evaluation.execution_resume import CheckpointOfferingForward
            path = CheckpointOfferingForward(path, execution_resume, directory / "execution_resume")
        forward = RecordedForward(path, directory / "module_invocations" / "executor")
        runtime.orchestrator.forward_path = forward
        return RecordedScenario(replace(runtime, forward_path=forward), directory)

    def _revision_runtime(self, scenario, root):
        from x_editor.evaluation.iterative_runtime import IterativeScenarioRuntime
        from x_editor.revision.deferred_terminal_design import deferred_module_provider, deferred_terminal_provider
        from x_editor.revision.refinement_cache import EpisodeReviewCache
        from x_editor.revision.refinement_reviewer import PublicRefinementReviewer
        from x_editor.revision.refinement_scheduling import judge_args, refinement_policy

        spec = deepcopy(self.config.revision)
        if spec.get("review_strategy") != "decision_first" or spec.get("nondelivery_review") != "raw_errors":
            raise ValueError("sequential revision requires raw-error retry, then decision-first review")
        for key, stage in (("deferred_module_designs", "modules"),
                           ("deferred_terminal_designs", "terminal")):
            # Both stages use shared.json internally, so their namespaces must
            # remain separate even for the same scenario and model settings.
            spec[key] = {"root": str(root / "public_designs" / stage), "identity": scenario.fingerprint}
        args = judge_args(spec)
        args.feedback_branch = "joint"
        audio_config = spec["audio_judge"]
        cache = EpisodeReviewCache()
        reviewer = PublicRefinementReviewer(module_designs={}, utility_design=None,
            args=args, audio_config=audio_config, cache=cache)
        reviewer.module_design_provider = deferred_module_provider(scenario, spec, args, audio_config)
        selector_args = copy(args)
        selector_args.feedback_branch = None
        selector_args.execution_purpose_review = False
        selector_args.strict_feedback_isolation = False
        selector = PublicRefinementReviewer(module_designs={}, utility_design=None,
            args=selector_args, audio_config=audio_config, cache=EpisodeReviewCache())
        selector.decision_cache = cache
        if spec.get("reuse_scope_preparation", False):
            selector.scope_preparation_cache = cache
        selector.module_design_provider = reviewer.module_design_provider
        return IterativeScenarioRuntime(factory=self._factory, reviewer=reviewer,
            selection_reviewer=selector, policy=refinement_policy(spec, "joint_feedback"),
            utility_design_provider=deferred_terminal_provider(scenario, spec, args, audio_config))

    async def arun(self, scenario: WorkflowRuntimeScenario | dict, *, output: str | Path,
                   mode: Literal["forward", "revision"] = "revision") -> dict:
        from x_editor.evaluation.iterative_runtime import delivered_artifact
        from x_editor.revision.module_chain_transport import recorded_http

        if mode not in ("forward", "revision"):
            raise ValueError("mode must be forward or revision")
        scenario = bind_assets(WorkflowRuntimeScenario.model_validate(scenario))
        root = Path(output).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=False)
        _write(root / "input.json", scenario.model_dump(mode="json"))
        _write(root / "runtime.json", self.config.model_dump(mode="json"))
        try:
            # This transport records requests; it does not change generation settings.
            with tool_selection_context(self.config.tools), recorded_http():
                if mode == "forward":
                    runtime = self._factory(root / "forward")
                    result = await runtime.arun_scenario(scenario, arm="W1", output_dir=root / "forward")
                    _write(root / "forward" / "scenario_result.json", result.model_dump(mode="json"))
                    receipt = {"mode": mode, "selected_artifact": delivered_artifact(result, scenario),
                               "selected_result": result.model_dump(mode="json")}
                else:
                    runtime = self._revision_runtime(scenario, root)
                    receipt = await runtime.arun_scenario(scenario, output_dir=root / "revision", utility_design=None)
                    receipt = {"mode": mode, **receipt}
            receipt["delivered"] = receipt.get("selected_artifact") is not None
            _write(root / "result.json", receipt)
            return receipt
        except Exception as error:
            _write(root / "error.json", {"type": type(error).__name__, "message": str(error)})
            raise

    def run(self, scenario, **kwargs) -> dict:
        return asyncio.run(self.arun(scenario, **kwargs))

    def edit(self, audio: str | Path, instruction: str, *, references=(), **kwargs) -> dict:
        assets = [{"asset_ref": "source", "path": str(audio), "role": "source", "order": 0}]
        assets += [{"asset_ref": f"reference{i}", "path": str(p), "role": "reference", "order": i + 1}
                   for i, p in enumerate(references)]
        return self.run({"audio_assets": assets, "turns": [{"turn_index": 0, "user_request": instruction,
            "source_asset_refs": ["source"], "reference_asset_refs": [a["asset_ref"] for a in assets[1:]]}]}, **kwargs)
