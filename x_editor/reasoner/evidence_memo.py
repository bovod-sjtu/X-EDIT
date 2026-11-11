"""Natural-language reasoning with a small, durable observation envelope.

This is an explicit experimental strategy in the shared runtime. It does not
select tools, repair model output, or infer observation needs from keywords.
"""

from __future__ import annotations

import copy
import json
from hashlib import sha256
from typing import Literal

from pydantic import Field

from x_editor.core.schemas import StrictModel
from x_editor.reasoner.contracts import (
    ObservationRequirement,
    ReasonerCallRequest,
    SemanticEditDraft,
    SemanticReasoningMemo,
)
from x_editor.reasoner.evidence_obligations import evidence_audio_scopes, reconcile_observations
from x_editor.reasoner.observation_review import (
    EVIDENCE_REVIEW_POLICY,
    TARGET_SELECTION_POLICY,
    source_observation_comparisons,
)
from x_editor.reasoner.openai_compatible import (
    OpenAICompatibleProseMemoSemanticReasoner,
    OpenAICompatibleReasonerContractError,
    _clear_client_json_invocations,
    _memo_call_error,
    _semantic_memo_client_for_phase,
    prose_semantic_memo_current_call_prompt,
    prose_semantic_memo_system_prompt,
)


class EvidenceMemoOutput(StrictModel):
    content: str = Field(min_length=1, max_length=4096)
    observation_ledger: list[ObservationRequirement] = Field(max_length=16)


class DraftMemoOutput(EvidenceMemoOutput):
    semantic_drafts: list[SemanticEditDraft] = Field(max_length=16)


class ObservationFinding(StrictModel):
    need_id: str = Field(min_length=1, max_length=128)
    status: Literal["pending", "resolved", "not_detected", "conflicting"]
    evidence_refs: list[str] = Field(max_length=32)
    finding: str = Field(max_length=2048)


class EvidenceReviewOutput(StrictModel):
    observation_findings: list[ObservationFinding] = Field(max_length=16)
    additional_observations: list[ObservationRequirement] = Field(max_length=16)
    content: str = Field(min_length=1, max_length=4096)


class DraftReviewOutput(EvidenceReviewOutput):
    semantic_drafts: list[SemanticEditDraft] = Field(max_length=16)


def _output_model(request: ReasonerCallRequest, include_drafts: bool):
    if request.phase == "evidence_review":
        return DraftReviewOutput if include_drafts else EvidenceReviewOutput
    return DraftMemoOutput if include_drafts else EvidenceMemoOutput


def _finding_alternatives(definition: dict, evidence_ids: list[str]) -> dict:
    pending = copy.deepcopy(definition)
    pending["required"] = list(pending["properties"])
    pending["properties"]["status"] = {"const": "pending", "type": "string"}
    alternatives = [pending]
    if evidence_ids:
        finding = copy.deepcopy(pending)
        finding["properties"]["status"] = {
            "enum": ["resolved", "not_detected", "conflicting"],
            "type": "string",
        }
        finding["properties"]["evidence_refs"] = {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "items": {"type": "string", "enum": evidence_ids},
        }
        alternatives.append(finding)
        pending["properties"]["evidence_refs"]["items"]["enum"] = evidence_ids
    else:
        pending["properties"]["evidence_refs"]["maxItems"] = 0
    return {"anyOf": alternatives}


def evidence_memo_response_schema(
    request: ReasonerCallRequest, *, include_drafts: bool = False
) -> dict:
    """Questions are defined once; reviews return updates by ID, not text copies."""
    schema = _output_model(request, include_drafts).model_json_schema()
    new_field = (
        "additional_observations" if request.phase == "evidence_review" else "observation_ledger"
    )
    schema["properties"][new_field]["items"] = _finding_alternatives(
        schema["$defs"]["ObservationRequirement"], list(request.reviewed_evidence_ids)
    )
    if request.phase == "evidence_review":
        prior = request.prior_memo.observation_ledger if request.prior_memo else []
        findings = schema["properties"]["observation_findings"]
        definition = schema["$defs"]["ObservationFinding"]
        if prior:
            definition["properties"]["need_id"]["enum"] = [item.need_id for item in prior]
        findings["items"] = _finding_alternatives(definition, list(request.reviewed_evidence_ids))
        findings["minItems"] = len(prior)
        findings["maxItems"] = len(prior)
    return schema


def _reviewed_requirements(output: EvidenceReviewOutput, request: ReasonerCallRequest):
    previous = {item.need_id: item for item in request.prior_memo.observation_ledger}
    updates = []
    for finding in output.observation_findings:
        if finding.need_id not in previous:
            raise ValueError("observation finding references an unknown need")
        updates.append(
            previous[finding.need_id].model_copy(update=finding.model_dump(exclude={"need_id"}))
        )
    if any(item.need_id in previous for item in output.additional_observations):
        raise ValueError("new observation needs require new IDs")
    return [*updates, *output.additional_observations]


_OBSERVATION_POLICY = TARGET_SELECTION_POLICY + """Your responsibility is to identify facts required for the requested edit.
If such an audio fact is missing, declare a pending observation with a precise
fact_kind question, audio_refs, required temporal precision, and why it matters.
The workflow will request Perception and return its observations to you. Do not
infer absence from an ASR omission, unmeasured property, partial coverage, or a
service failure. Distinguish a user's unclear intent from an unobserved audio fact.
When the requested sound is absent from a transcript, ask for direct listening
over the relevant audio, including background and non-transcribed regions. Ask
what is audible and where, not whether an ASR candidate happens to match. Use
local listening to refine a discovered region when its boundary matters.
When a requested word disagrees with ASR, request contextual re-recognition or
direct audio review. Words from the user's request and plausible readings are
hypotheses to check, not supplied truth. Do not reinterpret the user's edit or
declare the word absent without that check. Alignment of proposed text does not
establish that the words were spoken.
Different admitted speaker tags identify different observed speakers; names inside
spoken content do not create speakers. Resolve ordinal selectors by observed first
speech time. Infer name-to-speaker relationships only from supported context.
Overlapping speaker intervals can describe simultaneous speech, including chorus;
they do not by themselves contradict one another. Distinguish observed coactivity,
unknown word ownership, and incompatible claims about the same speech event.
Request attribution only where the edit actually depends on it. Identifying a
sound's interval does not establish that it can be edited alone in a mixture.
First distinguish sequential sounds from simultaneous ones: two observed intervals
overlap only when they intersect on the same audio timeline. Several voices in
one file are not evidence that they sound together. A candidate containing an
event supplies a search window, not necessarily the event's exact boundaries;
keep directly observed event boundaries distinct from ASR sentence bounds.
Preserve the user's target extent and other simultaneous sounds. Do not invent
a separation requirement for voices that occur entirely outside the target region.
For selective event removal, distinguish the event interval from the components
inside it: bind target and protected speech/background observations to the same
asset version. Use supplied word/speech intervals when they resolve coactivity;
an un-timestamped transcript does not establish when speech begins or ends. If
coactivity is unknown and affects preservation, ask whether protected sound is
audible within that specific interval. Do not resolve this from event presence
alone or infer that speech starts where the proposed event region ends.
If this non-overlap claim comes only from a model-only caption marked needs_review,
it does not yet justify muting/deleting the mixture. Request independent speech
timing (speech_context with word_timestamps/word precision) or targeted listening
that explicitly checks the protected component. Reuse such evidence if already
available. Record conflicting speech/event timing instead of treating a fluent
caption such as 'before speaking' as a confirmed empty-speech interval.
A supported event detection supports that event, not the absence of speech or
other sources. If ASR reports speech but a caption says speech is absent, retain
the conflict and request same-version speech timing before destructive interval
processing, even when the event itself has strong expert support. If only part
of the event can be removed without touching protected speech, keep the remaining
part explicitly unresolved; shortening the target span does not complete removal.
On review, return observation_findings with only need_id, status, evidence_refs,
and finding for every prior question. Do not copy or rewrite the question definition;
the runtime retains it. Add additional_observations only for newly discovered needs.
Cite the supplied evidence IDs that actually support the answer.
Mark resolved only when the observation actually resolves the required property
and scope at the granularity needed by this edit; receiving a report alone does
not resolve a question. Preserve conflicts between observations instead of treating
one report as a final verdict. Mark not_detected only for explicit negative findings from suitable
observation covering that scope. A transcript alone does not resolve voice gender
or events outside speech recognition. Keep conflicting/partial evidence pending or
conflicting and explain the next useful observation. Do not repeat an already
answered question. Missing evidence stays unknown, never a fabricated negative.
Do not invent a requirement for perfectly exact acoustic timestamps. For an
identified spoken target, admitted word alignment is usable boundary evidence
unless a concrete observation shows wrong wording, truncation, or misalignment.
The method's name or the ordinary uncertainty of a measurement alone is not a
reason to discard it and repeat the same question. Additional listening must
address a specific remaining ambiguity, with a capability that can answer it.
Declare semantic observations needed to identify the requested object or effect,
not every low-level measurement a possible tool might use. An unambiguous occurrence
identified in a supplied transcript can already identify a lexical target; do not
require external word timestamps solely to draft a text-conditioned edit. The
Planner owns execution prerequisites of its selected tool and can request any
additional measurement it actually needs. Only declare facts required for this
edit: do not demand speaker gender for an
ordinal selection, or invent requirements belonging to a different editing method.
For multiple assets, bind every observation to the exact audio_refs of the facts
being requested. A need_id or a prose label is not an audio binding. Keep the
reference property distinct from the content or region to edit. A comparison may
need separate observations of each asset; do not attach every available asset to
every need. Disputed wording is not automatically a prerequisite for a whole-asset
voice conversion that does not consume text; it remains relevant to later content
preservation checks. A delivery-only request does not establish that its eventual
tool is text-free. Preserve disputed wording and incomplete timing explicitly;
identifying the speaker or region does not resolve those conflicts. The Planner
must resolve text required by its selected synthesis or editing operation before
execution, including emotion and prosody operations that consume a transcript.
Keep user-authorized target scope and preservation constraints. A complete inventory
is not proof of which candidate satisfies a selector; an unresolved target never
authorizes editing all candidates. Review current evidence before copying an old
missing-fact statement. Preserve the user goal while correcting factual inferences.
Return concise natural-language content in the JSON envelope. The initial
observation_ledger is empty if no required observation is missing. On every
review return findings for ALL prior needs, including resolved ones, with
updated status and evidence citations. An empty list never means a prior need was
resolved. Once the supplied observations resolve a need, mark it resolved instead
of copying its stale pending finding. Do not choose
operations or tool parameters. The envelope is not a requested-change ledger.
"""


class OpenAICompatibleEvidenceMemoReasoner(OpenAICompatibleProseMemoSemanticReasoner):
    include_drafts = False

    def reason(self, request: ReasonerCallRequest, *, common_input: dict | None = None) -> SemanticReasoningMemo:
        self.strategy = "draft_memo" if self.include_drafts else "evidence_memo"
        self.last_invocation_trace = None
        self.last_semantic_memo = None
        self._provider_invocations = []
        system = prose_semantic_memo_system_prompt(request.phase, observation_envelope=True)
        system += "\n\n" + (
            EVIDENCE_REVIEW_POLICY if request.phase == "evidence_review" else _OBSERVATION_POLICY
        )
        if self.include_drafts:
            system += "\n" + _DRAFT_POLICY
        payload = json.loads(prose_semantic_memo_current_call_prompt(
            request, retry_context_mode=self.retry_context_mode))
        payload["response"] = (
            "Review JSON: observation_findings, additional_observations, then content."
            if request.phase == "evidence_review"
            else "Initial JSON: content, observation_ledger."
        )
        payload["pending_observation_history"] = [
            item.model_dump(mode="json")
            for item in (request.prior_memo.observation_ledger if request.prior_memo else [])
        ]
        from x_editor.reasoner.retry_context import share_retry_observation_ledger
        share_retry_observation_ledger(payload)
        payload["admitted_evidence_ids"] = list(request.reviewed_evidence_ids)
        if request.phase == "evidence_review":
            payload["source_observation_comparisons"] = source_observation_comparisons(request)
        audio_scopes = evidence_audio_scopes(request)
        payload["evidence_provenance"] = {
            "field_sources": request.evidence_field_sources,
            # This Reasoner cites observation IDs, never executable field paths.
            # Runtime retains the complete path inventories for admission. An
            # exact source/audio index supplies the role-level citation scope
            # without repeating every deeply nested path in the model prompt.
            "evidence_audio_scopes": {
                ref: sorted(audio_scopes.get(ref, set()))
                for ref in request.reviewed_evidence_ids
            },
        }
        if common_input is not None:
            # A controlled adviser sees exactly the Planner's public input, not
            # an additional projection or its own accumulated private history.
            # The role/schema differ; the supplied facts do not.
            from copy import deepcopy
            payload = deepcopy(common_input)
            system += (
                "\nThe supplied object is the common public Planner input. Its Planner "
                "output contract describes the downstream consumer, not your response. "
                "Return only the independent semantic memo required by your response "
                "schema. Tool contracts are available context, not instructions to emit a plan."
            )
        schema = evidence_memo_response_schema(request, include_drafts=self.include_drafts)
        current = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.last_prompt_package_fingerprint = sha256(
            (system + current + json.dumps(schema, sort_keys=True)).encode()
        ).hexdigest()
        client = _semantic_memo_client_for_phase(
            self.client,
            phase=request.phase,
            evidence_review_max_tokens=self.evidence_review_max_tokens,
        )
        self.last_prompt_budget = {
            "phase": request.phase,
            "provider_output_mode": "json_schema",
            "system_chars": len(system),
            "current_call_chars": len(current),
            "total_chars": len(system) + len(current),
            "strategy": self.strategy,
            "retry_context_mode": self.retry_context_mode,
        }
        _clear_client_json_invocations(client)
        try:
            raw = client.call_json(
                system,
                current,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "EvidenceMemoOutput", "schema": schema, "strict": True},
                },
                repair_schema_name="EvidenceMemoOutput",
            )
        except Exception as exc:
            error = _memo_call_error(client, exc)
            self._append_provider_invocations(
                succeeded=False, error=error, purpose="semantic_evidence_memo", client=client
            )
            self._set_invocation_trace(status="failed")
            raise error from exc
        self._append_provider_invocations(
            succeeded=True, purpose="semantic_evidence_memo", client=client
        )
        try:
            output = _output_model(request, self.include_drafts).model_validate(raw)
            drafts = list(getattr(output, "semantic_drafts", []))
            for draft in drafts:
                if draft.audio_ref not in request.source_audio_refs + request.reference_audio_refs:
                    raise ValueError("draft audio is outside current inputs")
                if not set(draft.evidence_refs).issubset(request.reviewed_evidence_ids):
                    raise ValueError("draft cites evidence outside current observations")
            observations = (
                _reviewed_requirements(output, request)
                if request.phase == "evidence_review"
                else output.observation_ledger
            )
            ledger = reconcile_observations(observations, request)
            memo = SemanticReasoningMemo(
                request_id=request.request_id,
                phase=request.phase,
                requested_changes=[],
                requested_change_ledger_status="provider_omitted",
                content=output.content,
                observation_ledger=ledger,
                semantic_drafts=drafts,
                remaining_execution_request=request.user_request,
                prior_memo_fingerprint=request.prior_memo.fingerprint
                if request.prior_memo
                else None,
            )
        except ValueError as exc:
            self._set_invocation_trace(status="failed")
            raise OpenAICompatibleReasonerContractError(str(exc)) from exc
        self.last_semantic_memo = memo
        self._set_invocation_trace(status="completed")
        return memo


_DRAFT_POLICY = """Also provide semantic_drafts when a content or sequence draft can reduce
ambiguity for the Planner. Draft the intended result in natural language; when
helpful annotate the observed transcript with pseudotags such as [replace old → new].
Separate source context from the smallest region whose content should change.
Keep unchanged words outside edit markers unless the user's goal requires their
resynthesis. An explicitly requested whole-utterance transformation remains global.
Identify the particular occurrence when wording repeats, retain independent edits,
and state preservation requirements. Do not guess missing source text, measured
boundaries, identities, or tool parameters. A draft is a revisable proposal, not
source evidence or a mandatory service syntax. Correct drafts after new evidence;
never let a stale draft override the user's goal. Return semantic_drafts=[] when
no useful supported draft can be made. Do not mechanically trim common prefixes.
"""


class OpenAICompatibleDraftMemoReasoner(OpenAICompatibleEvidenceMemoReasoner):
    include_drafts = True
