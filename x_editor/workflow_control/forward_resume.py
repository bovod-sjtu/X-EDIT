"""Explicit, bounded reuse of a mechanically accepted failed-execution prefix.

This is not a quality rollback policy. The caller keeps the original failure and
judgment history; only byte-bound products can be reused, never stale runtime
state. Ineligible requests execute the complete repaired plan once.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Any
from pathlib import Path
import hashlib
from x_editor.runtime.workflow_case import RuntimeAudioBinding
from pydantic import Field, model_validator
from x_editor.core.schemas import AudioEditPlan
from x_editor.workflow_control.schemas import SuffixResumeExecutionRequest, WorkflowFailureRegistry
from x_editor.workflow_control.suffix_resume import SuffixResumeAdmissionGate, SuffixResumeAuditor, full_replay_admission
if TYPE_CHECKING:
    from x_editor.runtime.forward_execution import ForwardExecutionResult


class RestrictedForwardResume(SuffixResumeExecutionRequest):
    protocol: Literal['x-editor.restricted-forward-resume/v1'] = 'x-editor.restricted-forward-resume/v1'
    prior_run_id: str = Field(min_length=1)
    source_bindings: list[RuntimeAudioBinding] = Field(default_factory=list)
    session_state: dict[str, Any] | None = None
    previous_attempt: int = Field(ge=1, le=2)
    max_total_attempts: int = Field(default=3, ge=2, le=3)

    @model_validator(mode='after')
    def budget_remaining(self):
        if self.previous_attempt >= self.max_total_attempts:
            raise ValueError('Resume consumes a new attempt; exhausted budget cannot be reset')
        return self


def request_from_failure(result: ForwardExecutionResult, *, previous_attempt: int,
                         max_total_attempts: int = 3,
                         legacy_audio_bindings: list[RuntimeAudioBinding] | None = None) -> RestrictedForwardResume:
    if result.status == 'completed' or result.executed_plan is None or result.artifact_audit is None:
        raise ValueError('Restricted resume requires an executed failure with an artifact audit')
    plan = result.executed_plan
    registry = WorkflowFailureRegistry(run_id=result.run_id, status='unresolved', run_status='failed',
        terminal_failed_step_ids=[s.id for s in plan.steps if s.status == 'failed'],
        unresolved_failure_codes=result.failure_codes)
    checkpoint = SuffixResumeAuditor().audit(plan, result.artifact_audit, registry)
    bindings = result.runtime_audio_bindings or legacy_audio_bindings or []
    by_ref = {b.audio_ref: b for b in bindings}
    for source in checkpoint.source_artifacts:
        binding = by_ref.get(source.path)
        if binding is None or not binding.content_sha256:
            continue  # Missing historic identity remains ineligible.
        path = Path(binding.materialized_path)
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != binding.content_sha256:
            continue
        source.path = str(path)
        source.size_bytes = path.stat().st_size
        source.content_sha256 = binding.content_sha256
    return RestrictedForwardResume(checkpoint=checkpoint, original_plan=plan.model_dump(mode='json'),
        prior_run_id=result.run_id, source_bindings=bindings, session_state=result.recovery_session_state, previous_attempt=previous_attempt, max_total_attempts=max_total_attempts)


def admit_forward_resume(request: RestrictedForwardResume, candidate: AudioEditPlan,
                         current_bindings: list[RuntimeAudioBinding] | None = None, *,
                         session_state: dict[str, Any] | None = None):
    original = AudioEditPlan.model_validate(request.original_plan)
    if request.session_state != session_state:
        return full_replay_admission(candidate, reason_code='active_session_changed',
            plain_summary='Active session/version history changed; prior prefix cannot be presumed valid.')
    current = {b.audio_ref: b for b in current_bindings or []}
    for prior in request.source_bindings:
        now = current.get(prior.audio_ref)
        try:
            actual = hashlib.sha256(Path(now.materialized_path).read_bytes()).hexdigest() if now else None
        except OSError:
            actual = None
        if (not prior.content_sha256 or actual != prior.content_sha256 or now is None
                or (now.content_sha256, now.role, now.materialized_path)
                != (prior.content_sha256, prior.role, prior.materialized_path)):
            return full_replay_admission(candidate, reason_code='runtime_audio_binding_changed',
                plain_summary='A source or reference binding changed across attempts.')
    if original.request != candidate.request:
        return full_replay_admission(candidate, reason_code='original_goal_changed',
            plain_summary='A changed original request cannot inherit an accepted prefix.')
    # The v1 executor restores only output_audio_path. Other named outputs must
    # not silently disappear from a multi-track dependency graph.
    for step_id in request.checkpoint.reusable_step_ids:
        artifacts = [a for a in request.checkpoint.reusable_artifacts if a.producer_step_id == step_id]
        if len({a.path for a in artifacts}) != 1:
            return full_replay_admission(candidate, reason_code='multi_output_reuse_not_supported',
                plain_summary='This bounded protocol cannot restore all outputs of a multi-track step.')
    return SuffixResumeAdmissionGate().audit(request.checkpoint, original, candidate)
