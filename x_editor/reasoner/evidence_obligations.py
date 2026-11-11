"""Track observation obligations without interpreting user wording or audio."""

from __future__ import annotations

import json
from typing import Any

from x_editor.core.text_equivalence import equivalent_text_copy
from x_editor.reasoner.contracts import ObservationRequirement, ReasonerCallRequest


class ObservationDispatchError(ValueError):
    """Expose the contradictory handoff without guessing which asset was meant."""

    def __init__(self, message: str, *, needs: list, jobs: list) -> None:
        self.feedback = {
            "code": "observation_dispatch_scope_conflict",
            "required_observations": [n.model_dump(mode="json") for n in needs],
            "proposed_jobs": [
                {"job_id": getattr(j, "job_id", None), "primary_audio_ref": j.primary_audio_ref,
                 "supporting_audio_refs": list(j.supporting_audio_refs),
                 "observation_need_ids": list(j.observation_need_ids)} for j in jobs
            ],
            "next_attempt_guidance": (
                "Reasoner: verify each observation's scope against the public request and asset inventory; "
                "a need's label is not an asset binding. Planner: cover the corrected required assets with "
                "explicit observation jobs. Do not drop an asset or reinterpret an ID to bypass this check."
            ),
        }
        super().__init__(message + "; handoff=" + json.dumps(self.feedback, ensure_ascii=False))


def evidence_audio_scopes(request: ReasonerCallRequest) -> dict[str, set[str]]:
    """Join provenance by exact reviewed path, never by text inside a path or ID."""
    scopes: dict[str, set[str]] = {}
    for path, evidence_ids in request.evidence_path_sources.items():
        audio_ref = request.evidence_path_audio_refs.get(path)
        if not audio_ref:
            continue
        for evidence_id in evidence_ids:
            scopes.setdefault(evidence_id, set()).add(audio_ref)
    return scopes


def _pending_with_finding(
    item: ObservationRequirement, finding: str, *, clear_evidence: bool = False
) -> ObservationRequirement:
    updates: dict[str, Any] = {"status": "pending", "finding": finding}
    if clear_evidence:
        updates["evidence_refs"] = []
    return item.model_copy(update=updates)


def reconcile_observations(
    observations: list[ObservationRequirement], request: ReasonerCallRequest
) -> list[ObservationRequirement]:
    """Keep omitted obligations and validate scope and evidence provenance.

    Finding semantics are the Reasoner's responsibility. This checks references,
    never whether a transcript contains the name of a sound or person.
    """
    previous = {
        item.need_id: item
        for item in (request.prior_memo.observation_ledger if request.prior_memo else [])
    }
    merged = dict(previous)
    known_audio = set(request.source_audio_refs + request.reference_audio_refs)
    known_evidence = set(request.reviewed_evidence_ids)
    evidence_scopes = evidence_audio_scopes(request)
    seen: set[str] = set()
    for item in observations:
        if item.need_id in seen:
            raise ValueError("observation need IDs must be unique")
        seen.add(item.need_id)
        prior = previous.get(item.need_id)
        if not set(item.audio_refs).issubset(known_audio) and (
            prior is None or set(item.audio_refs) != set(prior.audio_refs)
        ):
            raise ValueError("observation audio scope is outside this call")
        if not set(item.evidence_refs).issubset(known_evidence):
            previous_refs = set(prior.evidence_refs) if prior is not None else set()
            if not set(item.evidence_refs).issubset(known_evidence | previous_refs):
                raise ValueError("observation cites evidence outside this call")
        if prior is not None:
            # The explanatory reason may evolve without changing the question.
            for field in ("fact_kind", "audio_refs", "temporal_precision"):
                before, after = getattr(prior, field), getattr(item, field)
                same = (
                    (set(before) == set(after))
                    if field == "audio_refs"
                    else (
                        equivalent_text_copy(before, after)
                        if isinstance(before, str) and isinstance(after, str)
                        else before == after
                    )
                )
                if not same:
                    raise ValueError("preserve a need's definition; add a new need to refine it")
        merged[item.need_id] = item
    # Apply current scope and provenance checks to retained as well as returned
    # entries. Missing metadata is an unresolved observation, not a model error.
    for key, item in merged.items():
        if not set(item.audio_refs).issubset(known_audio):
            merged[key] = _pending_with_finding(
                item,
                "The retained observation's audio scope is unavailable in this call; "
                "its finding cannot resolve a current target.",
            )
            continue
        if not set(item.evidence_refs).issubset(known_evidence):
            merged[key] = _pending_with_finding(
                item,
                "Prior evidence is outside the current observation inventory.",
                clear_evidence=True,
            )
            continue
        if item.status == "pending":
            continue
        if any(not evidence_scopes.get(ref) for ref in item.evidence_refs):
            merged[key] = _pending_with_finding(
                item,
                "Evidence-to-audio scope metadata is unavailable; the required "
                "observation remains unconfirmed.",
            )
            continue
        cited_audio = set().union(*(evidence_scopes[ref] for ref in item.evidence_refs))
        if not set(item.audio_refs).issubset(cited_audio):
            merged[key] = _pending_with_finding(
                item,
                "The cited evidence does not cover the observation's audio scope; "
                "the required observation remains unconfirmed.",
            )
    if len(merged) > 16:
        raise ValueError("observation ledger exceeds its declared bound")
    return list(merged.values())


def pending_observations(memo: Any) -> list[ObservationRequirement]:
    return [item for item in memo.observation_ledger if item.status in {"pending", "conflicting"}]


def validate_observation_dispatch(turn: Any, planner_input: Any) -> None:
    """Require explicit task coverage of declared unresolved observations."""
    pending = pending_observations(planner_input.semantic_memo)
    if not pending:
        return
    if turn.decision == "cannot_plan":
        reasons = getattr(turn, "cannot_plan_reasons", ())
        if reasons and all(
            getattr(reason, "code", None) in {"unsupported_capability", "missing_perception_route"}
            for reason in reasons
        ):
            return  # Stop explicitly; the memo's unanswered observations stay pending.
    if turn.decision != "request_perception":
        raise ValueError(
            "required observations remain unresolved; request Perception before execution"
        )
    needs = {item.need_id: item for item in pending}
    covered: dict[str, set[str]] = {key: set() for key in needs}
    for job in turn.perception_jobs:
        job_refs = {job.primary_audio_ref, *job.supporting_audio_refs}
        for key, item in needs.items():
            covered[key].update(set(item.audio_refs) & job_refs)
        for need_id in job.observation_need_ids:
            if need_id not in needs:
                raise ObservationDispatchError("Perception task references an unknown pending observation",
                                               needs=pending, jobs=turn.perception_jobs)
            need = needs[need_id]
            if not set(need.audio_refs).intersection(job_refs):
                raise ObservationDispatchError("Perception task does not cover the observation's audio scope",
                                               needs=pending, jobs=turn.perception_jobs)
    if any(set(item.audio_refs) != covered[key] for key, item in needs.items()):
        raise ObservationDispatchError("Perception tasks omit a required audio scope",
                                       needs=pending, jobs=turn.perception_jobs)
    # Need IDs on jobs are optional audit references, not semantic proof that a
    # question was answered. Only the next evidence review can resolve a need.
