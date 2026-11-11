"""Auditable runtime rewrite rule registry for ``AudioEditExecutor``.

The executor still owns the low-level mutation methods for now. This module
keeps the rule order and metadata out of the execution loop so each rewrite has
a stable id, rationale, source, and test ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from x_editor.core.schemas import AudioEditStep

RuntimeRewriteSource = Literal[
    "tool_boundary",
    "schema_contract",
    "planner_compat",
    "benchmark_metadata_repair",
    "legacy_compat",
]


@dataclass(frozen=True)
class RuntimeRewriteRule:
    rule_id: str
    method_name: str
    rationale: str
    source: RuntimeRewriteSource
    applies_when: str
    rewrites: str
    tests: tuple[str, ...]
    enabled_by_default: bool = True

    def record(self, *, step_id: str) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "step_id": step_id,
            "method": self.method_name,
            "source": self.source,
            "rationale": self.rationale,
            "applies_when": self.applies_when,
            "rewrites": self.rewrites,
            "tests": list(self.tests),
        }


def default_pre_gemini_rewrite_rules() -> tuple[RuntimeRewriteRule, ...]:
    return (
        RuntimeRewriteRule(
            rule_id="unsupported_nonhuman_voice_to_audio_effect",
            method_name="_maybe_rewrite_unsupported_nonhuman_voice_request",
            rationale="Non-human voice conversion is outside voice conversion tools; expose best-effort audio coloration.",
            source="tool_boundary",
            applies_when="unsupported request asks for animal/non-human vocal coloration",
            rewrites="unsupported -> audio_effects",
            tests=("test_unsupported_nonhuman_voice_request_falls_back_to_audio_effects",),
        ),
        RuntimeRewriteRule(
            rule_id="voice_order_swap_to_timeline_reorder",
            method_name="_maybe_rewrite_voice_order_swap_to_timeline_reorder",
            rationale="Speaker order swaps are timeline operations, not speech synthesis operations.",
            source="schema_contract",
            applies_when="request asks to swap/reorder existing speaker turns",
            rewrites="requested operation -> artifact_ops.reorder_spans",
            tests=("test_voice_order_swap_rewrites_to_timeline_reorder",),
        ),
        RuntimeRewriteRule(
            rule_id="cross_file_intra_voice_transfer_to_seed_vc",
            method_name="_maybe_rewrite_cross_file_intra_voice_transfer",
            rationale="Cross-file voice transfer should use SeedVC/reference VC instead of same-audio intra transfer.",
            source="tool_boundary",
            applies_when="intra_audio_voice_transfer request actually references another input audio",
            rewrites="intra_audio_voice_transfer -> seed_vc",
            tests=("test_cross_file_intra_audio_voice_transfer_rewrites_to_scoped_seed_vc",),
        ),
        RuntimeRewriteRule(
            rule_id="reference_free_tse_to_timeline_extract",
            method_name="_maybe_rewrite_reference_free_tse_to_timeline_extract",
            rationale="TSE requires reference enrollment; reference-free speaker extraction needs perception-grounded spans.",
            source="tool_boundary",
            applies_when="tse request lacks reference audio and describes a target speaker/voice in the same audio",
            rewrites="tse -> artifact_ops.extract_spans",
            tests=("test_reference_free_tse_speaker_description_rewrites_to_timeline_extract",),
        ),
        RuntimeRewriteRule(
            rule_id="unsupported_audio_effect_to_audio_effects",
            method_name="_maybe_rewrite_unsupported_audio_effect_request",
            rationale="Known acoustic/channel effects should route to audio_effects instead of remaining unsupported.",
            source="planner_compat",
            applies_when="unsupported request describes a supported acoustic/channel effect preset",
            rewrites="unsupported -> audio_effects",
            tests=("test_whisper_style_rewrites_to_audio_effects_instead_of_boundary_failure",),
        ),
    )


def default_post_gemini_rewrite_rules() -> tuple[RuntimeRewriteRule, ...]:
    return (
        RuntimeRewriteRule(
            rule_id="emphasis_pitch_rate_to_p1_local_resynthesis",
            method_name="_maybe_rewrite_emphasis_pitch_rate_to_p1_resynthesis",
            rationale="Emphasis/stress edits need local resynthesis and splice recovery instead of pitch-only DSP.",
            source="tool_boundary",
            applies_when="pitch_rate_edit request asks to emphasize or stress a word/phrase without explicit legacy strategy",
            rewrites="edit_data_pipeline.pitch_rate_edit -> emphasis_stress_edit",
            tests=("test_emphasis_pitch_rate_plan_upgrades_to_p1_local_resynthesis",),
        ),
    )


def absolute_timeline_input_binding_rule() -> RuntimeRewriteRule:
    """Return the deterministic base-audio binding invariant for span tools."""

    return RuntimeRewriteRule(
        rule_id="bind_absolute_timeline_spans_to_base_audio_v1",
        method_name="_maybe_use_base_audio_for_absolute_spans",
        rationale=(
            "Original-timeline coordinates must bind to the immutable base audio, "
            "not to a shortened or reordered intermediate artifact."
        ),
        source="schema_contract",
        applies_when=(
            "a span operation explicitly declares the base timeline, or its "
            "coordinates fit the base audio but not the current intermediate"
        ),
        rewrites="input_audio_path: current intermediate -> base input audio",
        tests=(
            "test_frozen_forward_path_binds_absolute_spans_without_repair_budget",
            "test_executor_respects_explicit_current_timeline_reference",
        ),
    )


def apply_runtime_rewrite_rules(
    *,
    executor: Any,
    step: AudioEditStep,
    state: dict[str, Any],
    rules: tuple[RuntimeRewriteRule, ...],
) -> None:
    for rule in rules:
        if not rule.enabled_by_default:
            continue
        before = _step_signature(step)
        method = getattr(executor, rule.method_name)
        result = method(step, state)
        after = _step_signature(step)
        if bool(result) or after != before:
            _record_rule_hit(rule, step=step, state=state)


def _step_signature(step: AudioEditStep) -> tuple[Any, Any, Any]:
    return (step.operation, dict(step.inputs), dict(step.params))


def _record_rule_hit(
    rule: RuntimeRewriteRule,
    *,
    step: AudioEditStep,
    state: dict[str, Any],
) -> None:
    record = rule.record(step_id=step.id)
    state.setdefault("runtime_rewrite_rule_hits", []).append(record)
    rewrite = (state.get("operation_rewrites") or {}).get(step.id)
    if isinstance(rewrite, dict):
        rewrite.setdefault("rule_id", rule.rule_id)
        rewrite.setdefault("rule_source", rule.source)
        rewrite.setdefault("rule_rationale", rule.rationale)


def record_runtime_rewrite_rule_hit(
    rule: RuntimeRewriteRule,
    *,
    step: AudioEditStep,
    state: dict[str, Any],
) -> None:
    """Record a rule applied outside the pre/post operation rewrite passes."""

    _record_rule_hit(rule, step=step, state=state)
