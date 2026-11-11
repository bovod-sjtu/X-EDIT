"""One provider call for interpretation and PlanIR, using the shared validator.

This is a benchmark-neutral architecture intervention. The two Python protocol
views below expose products from ONE response; they are not two LLM agents.
No independent Reasoner answer is supplied to the integrated model.
"""
from __future__ import annotations

import copy
import hashlib
import json

from x_editor.core.invocation_trace import ComponentInvocationTrace
from x_editor.planner._invocation_trace import provider_records_from_client
from x_editor.planner.memo_qualified import (
    OpenAICompatibleMemoQualifiedPlanner,
    bind_operation_contract_runtime_domains,
    memo_qualified_planner_provider_response_format,
    project_operation_capability_contracts,
)
from x_editor.planner.prompt_package import (
    _memo_qualified_planner_prompt_projection,
    _memo_qualified_planner_output_contract,
    integrated_planner_system_prompt,
    observation_input_contracts,
)
from x_editor.planner.reasoned_workflow import MemoQualifiedPlannerInput
from x_editor.planner.contract_projection import group_step_arguments
from x_editor.planner.argument_table import factor_argument_contracts
from x_editor.planner.prompt_package import project_memo_qualified_candidate_operation_contracts
from x_editor.reasoner.contracts import SemanticReasoningMemo
from x_editor.reasoner.evidence_memo import (
    EvidenceMemoOutput,
    EvidenceReviewOutput,
    _OBSERVATION_POLICY,
    _reviewed_requirements,
    evidence_memo_response_schema,
)
from x_editor.reasoner.evidence_obligations import reconcile_observations


INTEGRATED_ROLE = """You are the single reasoning-and-planning module for audio editing.
There is no external Reasoner memo. Interpret the original user request, resolve
semantic relations using the supplied observations and public history, choose
legal tool arguments, and emit PlanIR in this same response. Do not assume a
missing fact or treat an ASR omission as proof of absence. When an editing fact
is missing, request targeted Perception; returned observations will be available
on your next planning call. Preserve the edit's scope and unaffected content.
References below to a Reasoner or memo describe interpretation duties that YOU
now own; no independent semantic conclusion is given or required. All tool,
source-evidence, audio identity, Session Graph, and PlanIR rules still apply.
Output a JSON object with the requested fields only. No hidden answer is supplied.
"""


class IntegratedSemanticPlanner:
    """Generate both products together; retain the actual response for validation."""

    def __init__(self, *, client, model_name, provider_name, capability_cards,
                 perception_catalog, edit_operation_params, reasoning='summary',
                 evidence_index_format='expanded', argument_contract_format='expanded'):
        if reasoning not in {'summary', 'direct'}:
            raise ValueError('integrated reasoning must be summary or direct')
        self.client, self.model_name, self.provider_name = client, model_name, provider_name
        self.contracts = project_operation_capability_contracts(capability_cards)
        self.perception_catalog = copy.deepcopy(perception_catalog)
        self.edit_operation_params = copy.deepcopy(edit_operation_params)
        self.reasoning = reasoning
        self.evidence_index_format = evidence_index_format
        self.argument_contract_format = argument_contract_format
        self.last_invocation_trace = None
        self.pending = None
        self.capability_feedback = None

    def bind_capability_feedback(self, feedback):
        self.capability_feedback = copy.deepcopy(feedback)

    def _input(self, request):
        # A structural projection holder, never presented as a Reasoner answer.
        # Its content is the public request; the memo and authority fields are
        # removed before transport. A real semantic product is built afterwards.
        holder = SemanticReasoningMemo(
            request_id=request.request_id, phase=request.phase, requested_changes=[],
            requested_change_ledger_status='provider_omitted', content=request.user_request,
            remaining_execution_request=request.user_request,
        )
        value = MemoQualifiedPlannerInput(
            planner_request_id=request.request_id,
            phase='evidence_planning' if request.prior_memo is None else 'execution_planning',
            user_request=request.user_request, execution_request=request.user_request,
            semantic_memo=holder, source_text=request.source_text,
            source_audio_refs=request.source_audio_refs, reference_audio_refs=request.reference_audio_refs,
            audio_asset_facts=request.audio_asset_facts, prior_turn_contexts=request.prior_turn_contexts,
            session_state=request.session_state, reviewed_evidence_ids=request.reviewed_evidence_ids,
            reviewed_evidence=request.reviewed_evidence,
            evidence_path_sources=request.evidence_path_sources,
            evidence_path_audio_refs=request.evidence_path_audio_refs,
            perception_capability_feedback=self.capability_feedback,
            perception_capability_catalog=self.perception_catalog,
            candidate_operation_contracts=bind_operation_contract_runtime_domains(
                self.contracts, source_audio_refs=request.source_audio_refs,
                reference_audio_refs=request.reference_audio_refs,
                frozen_operation_params=self.edit_operation_params),
        )
        payload = _memo_qualified_planner_prompt_projection(
            value, evidence_index_format=self.evidence_index_format,
            argument_contract_format='expanded')
        # This interface can execute on its first call. A capability summary
        # without the argument grammar is not sufficient to author that call.
        # Evidence availability controls admission, not visibility of the tool
        # contract; neither advice nor a representation adapter may replace it.
        payload['candidate_operation_contracts'] = [
            group_step_arguments(item)
            for item in project_memo_qualified_candidate_operation_contracts(
                value.candidate_operation_contracts, phase='execution_planning')
        ]
        if self.argument_contract_format == 'argument-table/v1':
            payload = factor_argument_contracts(payload)
        payload['operation_contract_visibility_policy'] = (
            'Full argument contracts are available for planning in every round. '
            'Their visibility does not establish that required evidence exists. '
            'Execute only with supported arguments; request missing observations '
            'when needed. Independent guidance does not override tool syntax.'
        )
        for key in ('semantic_memo', 'semantic_authority', 'semantic_conflict_policy',
                    'memo_normalization_applied', 'private_reasoning_visible'):
            payload.pop(key, None)
        payload['observation_input_contracts'] = observation_input_contracts(self.perception_catalog)
        payload['planner_output_contract'] = _memo_qualified_planner_output_contract(
            phase=value.phase, state_only=False,
            evidence_index_format=self.evidence_index_format)
        payload['previous_integrated_interpretation'] = (
            {'content': request.prior_memo.content,
             'observation_ledger': [v.model_dump(mode='json') for v in request.prior_memo.observation_ledger]}
            if request.prior_memo else None)
        if request.future_attempt_guidance is not None:
            raise ValueError('integrated architecture study requires outer recovery disabled')
        return value, payload

    def reason(self, request):
        """Protocol view of the semantic product from the joint provider call."""
        self.pending = None
        self.last_invocation_trace = None
        value, payload = self._input(request)
        wire = memo_qualified_planner_provider_response_format(phase=value.phase)
        from x_editor.planner.decision_schema import decision_payload_schema
        wire['json_schema']['schema'] = decision_payload_schema(wire['json_schema']['schema'])
        schema = {'type': 'object', 'additionalProperties': False,
                  'properties': {}, 'required': []}
        if self.reasoning == 'summary':
            summary = evidence_memo_response_schema(request)
            schema['$defs'] = summary.pop('$defs', {})
            schema['properties']['interpretation'] = summary
            schema['required'].append('interpretation')
        schema['properties']['turn'] = wire['json_schema']['schema']
        schema['required'].append('turn')
        policy = (
            'First write interpretation.content: a concise decision rationale stating the target, '
            'relevant semantic derivations, scope and what current evidence supports. Maintain '
            'interpretation.observation_ledger: missing facts stay pending, observations need '
            'supplied evidence IDs. Preserve prior question definitions when updating their status. '
            'Then produce turn with the executable plan or targeted observation request.'
            if self.reasoning == 'summary' else
            'Output turn directly, with its ordinary brief decision_summary. No separate '
            'interpretation or reasoning report is requested. You still own all semantic decisions '
            'and may request Perception before planning; do not guess missing facts.'
        )
        if self.reasoning == 'summary':
            if request.phase == 'evidence_review':
                policy = (
                    'Write interpretation.content with your updated interpretation. '
                    'Return interpretation.observation_findings for each prior need_id, '
                    'and interpretation.additional_observations for newly discovered needs. '
                    'Then produce turn: execute only when its necessary facts are supported, '
                    'otherwise request targeted Perception. Keep unresolved facts explicit.'
                )
            policy += '\nThe observation fields below belong inside interpretation.\n' + _OBSERVATION_POLICY
        system = self.system_role() + '\n' + policy + '\n\n' + integrated_planner_system_prompt()
        current = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.last_prompt_budget = {'architecture': 'integrated', 'reasoning': self.reasoning,
            'system_chars': len(system), 'current_call_chars': len(current),
            'max_completion_tokens': int(getattr(self.client, 'max_tokens', 0) or 0)}
        self.last_prompt_package_fingerprint = hashlib.sha256((system+current).encode()).hexdigest()
        if hasattr(self.client, 'json_invocations'):
            self.client.json_invocations = []
        records = []
        try:
            raw = self.client.call_json(system, current, response_format={'type': 'json_schema',
                'json_schema': {'name': 'IntegratedPlanningDecision', 'schema': schema, 'strict': True}},
                repair_schema_name='IntegratedPlanningDecision')
            records = provider_records_from_client(self.client, provider=self.provider_name,
                model=self.model_name, planner_purpose='integrated_reasoning_and_plan',
                start_index=1, succeeded=True)
            expected = {'turn', 'interpretation'} if self.reasoning == 'summary' else {'turn'}
            if not isinstance(raw, dict) or set(raw) != expected or not isinstance(raw['turn'], dict):
                raise ValueError('integrated response has missing or unexpected fields')
            if self.reasoning == 'summary':
                output_model = EvidenceReviewOutput if request.phase == 'evidence_review' else EvidenceMemoOutput
                output = output_model.model_validate(raw['interpretation'])
                content = output.content
                observations = (_reviewed_requirements(output, request)
                    if request.phase == 'evidence_review' else output.observation_ledger)
                ledger = reconcile_observations(observations, request)
            else:
                content = raw['turn'].get('decision_summary') or request.user_request
                ledger = []
            memo = SemanticReasoningMemo(
                request_id=request.request_id, phase=request.phase, content=content,
                requested_changes=[], requested_change_ledger_status='provider_omitted',
                remaining_execution_request=request.user_request, observation_ledger=ledger,
                prior_memo_fingerprint=request.prior_memo.fingerprint if request.prior_memo else None)
            self.pending = (memo.fingerprint, copy.deepcopy(raw['turn']))
        except Exception as exc:
            if not records:
                records = provider_records_from_client(self.client, provider=self.provider_name,
                    model=self.model_name, planner_purpose='integrated_reasoning_and_plan',
                    start_index=1, succeeded=False, error=exc)
            self._trace('failed', records)
            raise
        self._trace('completed', records)
        return memo

    def system_role(self):
        return INTEGRATED_ROLE

    def _trace(self, status, records):
        self.last_invocation_trace = ComponentInvocationTrace.build(
            component='reasoner', component_name='IntegratedSemanticPlanner', route='provider',
            status=status, provider_invocations=records,
            metadata={'architecture': 'integrated', 'semantic_product_is_joint_response': True,
                      'separate_reasoner_calls': 0, 'prompt_budget': self.last_prompt_budget,
                      'prompt_package_fingerprint': self.last_prompt_package_fingerprint})


class IntegratedTurnCompiler:
    """Validate the SAME response through the production PlanIR boundary, no LLM."""
    def __init__(self, provider):
        self.provider = provider
        self.last_invocation_trace = None

    def plan(self, planner_input):
        pending, self.provider.pending = self.provider.pending, None
        if pending is None or pending[0] != planner_input.semantic_memo.fingerprint:
            raise ValueError('integrated turn must match its actual semantic product')
        validator = OpenAICompatibleMemoQualifiedPlanner(
            model_name=self.provider.model_name, provider_name=self.provider.provider_name,
            client=self.provider.client)
        try:
            return validator.validate_provider_output(planner_input, pending[1])
        finally:
            trace = validator.last_invocation_trace
            self.last_invocation_trace = ComponentInvocationTrace.build(
                component='planner', component_name='IntegratedTurnCompiler', route='deterministic',
                status=trace.status if trace else 'failed', provider_invocations=[],
                metadata={**(trace.metadata if trace else {}), 'architecture': 'integrated',
                          'source': 'same_integrated_provider_response', 'new_provider_calls': 0})
