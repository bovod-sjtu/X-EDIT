"""Readable comparisons of source observations for semantic review only.

Execution authority describes which values have complete bindings. It is not a
ranking of what an observer actually heard. This view keeps those concepts
separate without modifying the admitted evidence or choosing a winning claim.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from x_editor.reasoner.contracts import ReasonerCallRequest


def source_observation_comparisons(request: ReasonerCallRequest) -> list[dict[str, Any]]:
    """Expose already-sanitized competing values beside their bound counterpart.

    Local IDs in a reported value stay in the source's namespace. The result is
    Reasoner prompt material only: it never enters the executable path inventory.
    No input text, timestamp, source preference, or resolution is inferred here.
    """
    report = request.reviewed_evidence.get("perception_report")
    if not isinstance(report, Mapping):
        return []
    reports = [report]
    for entry in report.get("audio_reports") or []:
        if isinstance(entry, Mapping) and isinstance(entry.get("report"), Mapping):
            reports.append(entry["report"])
    known_refs = set(request.source_audio_refs + request.reference_audio_refs)
    known_evidence = set(request.reviewed_evidence_ids)
    comparisons: list[dict[str, Any]] = []
    for scoped_report in reports:
        for claim in scoped_report.get("competing_observations") or []:
            if not isinstance(claim, Mapping):
                continue
            audio_ref = claim.get("audio_ref_id")
            evidence_id = claim.get("evidence_id")
            canonical_ids = claim.get("canonical_evidence_ids") or []
            if (
                audio_ref not in known_refs
                or evidence_id not in known_evidence
                or not set(canonical_ids).issubset(known_evidence)
                or scoped_report.get("audio_ref_id") not in {None, audio_ref}
            ):
                continue
            try:
                observed = json.loads(claim["reported_value_json"])
            except (KeyError, TypeError, ValueError):
                continue  # Raw evidence remains visible, including malformed diagnostics.
            field = claim.get("field")
            comparisons.append({
                "audio_ref_id": audio_ref,
                "observation_kind": claim.get("observation_kind"),
                "field": field,
                "bound_observation": {
                    "evidence_refs": list(canonical_ids),
                    "value": copy.deepcopy(scoped_report.get(field)),
                    "binding_role": "admitted execution representation; fallible observation",
                },
                "other_observation": {
                    "evidence_refs": [evidence_id],
                    "operation": claim.get("operation"),
                    "value": observed,
                    "binding_role": "observed claim; execution binding not yet admitted",
                    "value_namespace": claim.get("value_namespace"),
                },
            })
    return comparisons


TARGET_SELECTION_POLICY = """Separate locating candidates from establishing the user's selection criterion.
Word timing, an occurrence list, or a source-component label establishes only that
reported fact; it does not establish a candidate's contextual role or eligibility.
When the distinction matters and current evidence does not resolve it, ask a
targeted question about that distinction, not another inventory of the candidates.
Do not replace a semantic criterion with the first occurrence or a positional default.
Reuse an explicit, scoped finding when it already resolves the criterion; do not
add listening calls merely because a task contains a selector. Keep unresolved
selection explicit if the available evidence or observation budget is insufficient.
"""


EVIDENCE_REVIEW_POLICY = TARGET_SELECTION_POLICY + """Review the observations before writing the updated memo.
The user request defines the goal. The prior memo and its findings are revisable
interpretations, not additional evidence. For each existing observation need:
1. Read what the supplied Perception reports actually observed, including source
   observation comparisons. Distinguish direct hearing, transcript hypotheses,
   alignment, candidate-match judgments, and failed or uncertain observations.
2. Decide what they establish about this need on the same audio and relevant
   scope. A targeted re-observation can correct an inadequate initial transcript.
   Canonical/admitted means execution references are bound; it does not make ASR
   infallible or discredit another observer. When targeted listening resolves the
   disputed wording, update the semantic conclusion and mark that fact resolved.
   Any still-needed execution binding is a separate prerequisite for Planner;
   do not request the same hearing just because the old bound text differs.
   A bound transcript_revision records an explicit re-recognition of this asset;
   review its current wording rather than reopening a question solely because
   a superseded ASR report still appears in the observation history.
3. If reports really conflict, describe the actual conflicting observations and
   the remaining question. A candidate labeled no/false with zero confidence,
   unresolved status, or a contradictory explanation does not prove absence.
   Distinguish a candidate failing to match from an adequate search not detecting
   the target. Mark not_detected only for a supported negative over the required
   scope. Missing evidence remains pending; conflicting evidence stays conflicting.
4. Give observation_findings for every previous need, citing the evidence that
   supports each finding. Then write content from these updated findings. Add a
   new need only for a genuinely new required fact. Do not copy stale absence or
   missing-evidence assertions after a report answered them.

Words in the user's request are hypotheses, not transcripts. Prefer a contextual
re-recognition question that lets the observer report the actual words, including
an alternative or no target. Do not manufacture corrected speech from spelling
similarity, linguistic plausibility, a hotword hint, or forced alignment alone.
Neither recency nor execution authority alone establishes correctness. Keep
different assets, revisions, occurrences and simultaneous speakers separate.
Overlapping intervals can describe simultaneous speech; require word ownership
only where the intended edit depends on it. Ordinary timestamp uncertainty alone
does not invalidate usable alignment. A containing interval is not necessarily
the target's extent, and identifying a sound does not prove it can be isolated.
Retain the user's scope, unedited content and requested effect. Do not choose
tools or parameters, weaken the task, or claim a session completion without proof.
Return the declared JSON envelope, without private chain-of-thought.
"""


OBSERVATION_PURPOSE_POLICY = """Separate three kinds of unfinished work:
First inspect supplied audio_asset_facts. Version-bound whole-asset peak and RMS
are already measured; do not ask a listener to guess them. Whole-asset statistics
cannot stand in for local or isolated-speaker measurements. RMS is unweighted
signal amplitude, not LUFS or perceptual loudness.
- Observable source facts: request listening only when the answer can change the
  target binding, the requested result, preservation, or a real tool prerequisite.
  State that purpose and the required asset/time scope. An empty or failed
  observation supplies no new answer; it does not invalidate earlier usable facts.
- Authorized execution choices: when the user specifies an effect but leaves its
  degree unspecified, describe that effect and let Planner choose and disclose a
  supported, proportionate parameter. A chosen speed/gain value is not a fact to
  discover in the recording. Preserve any explicit user value exactly.
- Material intent ambiguity: distinguish multiple plausible targets or meanings
  that listening cannot resolve. Explain the actual ambiguity; do not disguise it
  as an unavailable perception route or arbitrarily pick an occurrence.

Spelling alternatives that sound the same need not identify different spoken
occurrences. Decide from the user's context and observed timing whether a unique
spoken target is already bound; preserve the observed transcript for execution.
Do not ask audio to reveal inaudible orthography, infer unheard words, or declare
all similar spellings equivalent. Exact transcript agreement is required only
where an operation depends on lexical content, not for every acoustic change.
Failed verification, rejected review, and diagnostics describing a possible error
are disputed observations, not confirmations of speaker identity or absence.
A local negative cannot settle an asset-wide question. Inspect actual observed
and unobserved intervals; request remaining coverage within the declared budget
when it could answer an outstanding need. Never infer silence from missing ASR.
"""
