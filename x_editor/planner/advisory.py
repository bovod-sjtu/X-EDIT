"""Independent guidance with a common Planner interface and decision authority.

The optional adviser adds information, not a second plan or an extra admission
gate. Both arms use the same direct decision schema and ordinary validators.
"""
import hashlib
import json

from x_editor.core.invocation_trace import ComponentInvocationTrace
from x_editor.planner.integrated import IntegratedSemanticPlanner, IntegratedTurnCompiler
from x_editor.reasoner.openai_compatible import OpenAICompatibleReasonerError


ADVISORY_ROLE = """You are the audio-editing Planner. Interpret the public request,
use observations and history, and produce a brief decision_summary and PlanIR or
a targeted Perception request. You may reason freely; no separate interpretation
report or Reasoner memo is requested from you. Optional independent guidance is
advisory: check it against current evidence and tool contracts. You own the same
decisions whether guidance is present or absent. Missing evidence calls for
targeted observation, not guessing or declaring an event absent. All ordinary
asset, evidence, tool and PlanIR checks apply. Preserve unaffected content.
"""


class AdvisorySemanticPlanner(IntegratedSemanticPlanner):
    """Expose a common direct Planner with an optional independent adviser."""

    def __init__(self, *, adviser=None, adviser_input_scope='module_projection', **kwargs):
        super().__init__(reasoning='direct', **kwargs)
        self.adviser = adviser
        self.adviser_memo = None
        self.adviser_trace = None
        self.adviser_error = None
        self.adviser_request = None
        if adviser_input_scope not in {'module_projection', 'common_planner_input'}:
            raise ValueError('unsupported adviser input scope')
        self.adviser_input_scope = adviser_input_scope
        self.common_input_fingerprint = None

    def system_role(self):
        return ADVISORY_ROLE

    def _input(self, request):
        # Outer recovery starts a new attempt. Its diagnostics are suggestions,
        # never new source evidence or permission to bypass admission.
        value, payload = super()._input(request.model_copy(update={'future_attempt_guidance': None}))
        if request.future_attempt_guidance is not None:
            payload['recovery_guidance'] = {
                module: request.future_attempt_guidance.prompt_projection(module=module)
                for module in ('reasoner', 'planner')
            }
            payload['recovery_guidance_policy'] = (
                'These are diagnostics from previous attempts, not observations of the current '
                'source. Reconsider the indicated decisions against current facts and contracts. '
                'Preserve correct conclusions; do not reconstruct old failed reasoning. '
                'Unavailable final audio cannot be verified before execution.'
            )
        self.common_input_fingerprint = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.adviser_trace = None
        self.adviser_error = None
        self.adviser_request = None
        if request.prior_memo is None:
            self.adviser_memo = None
        if self.adviser is not None:
            advisory_request = request.model_copy(update={
                'prior_memo': self.adviser_memo,
                'phase': request.phase if self.adviser_memo is not None else 'initial_understanding',
            })
            try:
                if self.adviser_input_scope == 'common_planner_input':
                    advisory_request = request.model_copy(update={
                        'prior_memo': None, 'phase': 'initial_understanding'})
                    self.adviser_request = advisory_request.model_dump(mode='json')
                    self.adviser_memo = self.adviser.reason(advisory_request, common_input=payload)
                else:
                    self.adviser_request = advisory_request.model_dump(mode='json')
                    self.adviser_memo = self.adviser.reason(advisory_request)
            except OpenAICompatibleReasonerError as exc:
                # Advice is optional. Keep the failed call for audit, discard
                # its unusable state, and let the ordinary Planner decide from
                # the same public facts available to the unguided condition.
                self.adviser_memo = None
                self.adviser_error = {'type': type(exc).__name__, 'message': str(exc)}
            finally:
                self.adviser_trace = self.adviser.last_invocation_trace
            if self.adviser_memo is not None:
                payload['independent_guidance'] = self.adviser_memo.model_dump(mode='json')
        # The holder passed to the validator has no adviser ledger. Both arms
        # retain the same ability to request observations or emit executable IR.
        return value, payload

    def _trace(self, status, records):
        advice = self.adviser_trace.provider_invocations if self.adviser_trace else []
        self.planner_invocation_trace = ComponentInvocationTrace.build(
            component='planner', component_name='AdvisorySemanticPlanner', route='provider',
            status=status, provider_invocations=records,
            metadata={'architecture': 'common_planner_advisory',
                      'common_input_sha256': self.common_input_fingerprint,
                      'adviser_input_scope': self.adviser_input_scope,
                      'prompt_budget': self.last_prompt_budget,
                      'prompt_package_fingerprint': self.last_prompt_package_fingerprint})
        # The protocol calls reason() first, but that must not charge the
        # Planner's call against the Reasoner's one-call admission allowance.
        # Compiler below publishes the already-made Planner call, exactly once.
        self.last_invocation_trace = ComponentInvocationTrace.build(
            component='reasoner', component_name='IndependentAdviser',
            route='provider' if advice else 'deterministic',
            status=status, provider_invocations=advice,
            metadata={'architecture': 'common_planner_advisory',
                      'independent_guidance': self.adviser is not None,
                      'separate_reasoner_calls': len(advice),
                      'adviser_memo': self.adviser_memo.model_dump(mode='json') if self.adviser_memo else None,
                      'adviser_request': self.adviser_request,
                      'adviser_trace': self.adviser_trace.model_dump(mode='json') if self.adviser_trace else None,
                      'adviser_error': self.adviser_error,
                      'advice_used': self.adviser_memo is not None,
                      'planner_call_recorded_at': 'planner'})


class AdvisoryTurnCompiler(IntegratedTurnCompiler):
    """Validate the common decision and attribute its actual provider call."""

    def plan(self, planner_input):
        try:
            return super().plan(planner_input)
        finally:
            validation = self.last_invocation_trace
            actual = self.provider.planner_invocation_trace
            self.last_invocation_trace = ComponentInvocationTrace.build(
                component='planner', component_name='AdvisoryTurnCompiler', route='provider',
                status=validation.status if validation else 'failed',
                provider_invocations=actual.provider_invocations,
                metadata={**actual.metadata, 'validation': validation.metadata if validation else {},
                          'new_compiler_provider_calls': 0})
