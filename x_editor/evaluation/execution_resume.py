"""Opt-in failed-prefix reuse between ordinary, independently admitted attempts.

The Planner still runs normally. This adapter offers the previous failure's
checkpoint to the existing Forward admission gate; it never edits the new plan
or grants quality approval. Completed but unsatisfactory audio is replayed fully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib

from x_editor.workflow_control.forward_resume import (
    RestrictedForwardResume,
    request_from_failure,
)


@dataclass
class ExecutionResumeOffer:
    request: RestrictedForwardResume | None
    receipt: dict
    asset_hashes: dict[str, str] = field(default_factory=dict)


def _attest_bindings(bindings, asset_hashes):
    """Only pre-execution public asset hashes can fill an older missing hash."""
    result=[]
    for binding in bindings:
        item=binding.model_copy(deep=True)
        expected=asset_hashes.get(item.materialized_path)
        if expected and item.content_sha256 is None:
            if hashlib.sha256(Path(item.materialized_path).read_bytes()).hexdigest() == expected:
                item.content_sha256=expected
        result.append(item)
    return result


def _restore_recorded_identity(failed, directory, asset_hashes):
    """Read the same captured Forward call when an older result omitted bindings."""
    if directory is None:
        return failed, None
    from x_editor.runtime.forward_execution import ForwardExecutionRequest, ForwardExecutionResult
    for output in sorted((Path(directory)/'module_invocations/executor').glob('*/output.json')):
        saved=ForwardExecutionResult.model_validate_json(output.read_text())
        if saved.run_id != failed.run_id:
            continue
        if saved.model_dump(mode='json') != failed.model_dump(mode='json'):
            raise ValueError('Recorded Forward result differs from prior scenario')
        raw=json.loads((output.parent/'input.json').read_text())['args'][0]
        captured=ForwardExecutionRequest.model_validate(raw)
        if captured.plan.model_dump(mode='json') != failed.input_plan.model_dump(mode='json'):
            raise ValueError('Recorded Forward input plan differs from the failed attempt')
        return failed.model_copy(update={
            'runtime_audio_bindings':_attest_bindings(captured.audio_bindings,asset_hashes),
            'recovery_session_state':captured.session_state.model_dump(mode='json') if captured.session_state else None,
        }), str(output.parent)
    return failed, None


def prepare_execution_resume(previous, *, scenario_fingerprint, previous_attempt,
                             max_total_attempts, previous_directory=None, source_assets=()):
    """Account for every recovery attempt, including ineligible full replays."""
    if not 1 <= previous_attempt < max_total_attempts <= 3:
        raise ValueError("Restricted resume consumes the same total three-attempt budget")
    receipt = dict(policy="failed_prefix", previous_attempt=previous_attempt,
                   max_total_attempts=max_total_attempts, offered=False,
                   mechanical_reuse_not_quality_certification=True)
    if previous is None:
        return ExecutionResumeOffer(None, {**receipt, "reason": "no_structured_previous_result"})
    if previous.scenario_fingerprint != scenario_fingerprint or previous.assigned_arm != "W1":
        raise ValueError("Cannot reuse execution from another scenario or workflow")
    if previous.status == "completed":
        return ExecutionResumeOffer(None, {**receipt, "reason": "quality_rollback_not_qualified"})
    failures = [turn for turn in previous.turn_results
                if turn.result.forward_result is not None
                and turn.result.forward_result.status != "completed"]
    if len(failures) != 1:
        return ExecutionResumeOffer(None, {**receipt, "reason": "no_unique_failed_forward"})
    turn = failures[0]
    receipt["prior_turn_index"] = turn.turn_index
    hashes={asset.path:asset.content_sha256 for asset in source_assets}
    try:
        failed, capture=_restore_recorded_identity(turn.result.forward_result,previous_directory,hashes)
        receipt['recorded_forward_identity']=capture
        request = request_from_failure(
            failed, previous_attempt=previous_attempt,
            max_total_attempts=max_total_attempts,
        )
    except (ValueError, OSError) as exc:
        return ExecutionResumeOffer(None, {**receipt, "reason": "checkpoint_unavailable",
                                          "qualification_error": f"{type(exc).__name__}: {exc}"})
    receipt.update(prior_run_id=request.prior_run_id,
                   reusable_step_ids=request.checkpoint.reusable_step_ids,
                   checkpoint_status=request.checkpoint.status)
    if not request.checkpoint.reusable_step_ids:
        return ExecutionResumeOffer(None, {**receipt, "reason": "no_accepted_reusable_prefix"})
    return ExecutionResumeOffer(request, {**receipt, "offered": True,
                                          "reason": "requires_new_plan_and_dependency_admission"}, hashes)


class CheckpointOfferingForward:
    """Inject only the typed opt-in request; Forward retains all admission rights."""

    def __init__(self, component, offer: ExecutionResumeOffer, directory):
        self.component, self.offer = component, offer
        self.directory = Path(directory)
        self.call_index = 0

    def __getattr__(self, name):
        return getattr(self.component, name)

    async def arun(self, request, *, observer=None):
        index = self.call_index
        self.call_index += 1
        prior = self.offer.request
        receipt = {**self.offer.receipt, "forward_run_id": request.run_id}
        if request.suffix_resume is not None:
            raise ValueError("A Forward call cannot consume two execution checkpoints")
        if prior is not None and prior.original_plan["request"] == request.plan.request:
            request = request.model_copy(update={"suffix_resume": prior,
                "audio_bindings": _attest_bindings(request.audio_bindings,self.offer.asset_hashes)})
            receipt["attached"] = True
        else:
            receipt.update(attached=False, reason=("different_turn_goal" if prior else receipt["reason"]))
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f"{index:04d}.request.json").open("x") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
        result = await self.component.arun(request, observer=observer)
        decisions = [event.payload for event in result.events
                     if event.kind == "forward_suffix_resume_decision"]
        with (self.directory / f"{index:04d}.result.json").open("x") as stream:
            json.dump(dict(admissions=decisions, status=result.status,
                           tool_call_count=result.tool_call_count), stream,
                      ensure_ascii=False, indent=2)
        return result
