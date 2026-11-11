"""Production review support extracted without changing function bodies."""

import json


from x_editor.revision.atomic_module_review import public_context


from x_editor.revision.refinement_checkpoints import recorded_tool_observations


from x_editor.revision.refinement_cache import semantic_identity


from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint


from x_editor.evaluation.iterative_runtime import write_record


from x_editor.evaluation.regional_listening import LISTENING_PACKET_PROMPT


from x_editor.evaluation.refinement_observation import DESCRIBE_PROMPT, CHECK_PROMPT, WINDOW_COORDINATE_PROMPT


from x_editor.evaluation.runtime_scope_contracts import SCOPE_PLAN_PROMPT, SCOPE_COVERAGE_PROMPT


from x_editor.evaluation.runtime_scope_review import review_runtime_scopes


from x_editor.evaluation.scope_speech_observation import ScopeSpeechObserver, SpeechObservationConfig


def runtime_scope_evidence(result):
    """Use actual observations, never private development labels or gold spans.

Internal audio refs remain internal: no ordinal or basename identity guessing.
The planner must obtain a bound observation when a report cannot be related to
the supplied version. Tool reports are explicitly fallible.
    """
    evidence = {}
    if result is None:
        return evidence
    declared_ids = {}
    for turn in result.turn_results:
        value = turn.result
        for index, record in enumerate(getattr(value, 'perception_history', ())):
            record = record.model_dump(mode='json') if hasattr(record, 'model_dump') else record
            key = f'perception_{turn.turn_index}_{index}'
            evidence[key] = {
                'authority':'runtime_observation_not_ground_truth',
                'turn_index':turn.turn_index,
                **{k:record[k] for k in ('request','status','evidence') if k in record},
            }
            declared = (record.get('evidence') or {}).get('evidence_id')
            if declared:
                declared_ids.setdefault(declared, []).append(key)
        for index, observation in enumerate(recorded_tool_observations(value)):
            evidence[f'tool_{turn.turn_index}_{index}'] = observation
    # The original typed evidence ID is a valid citation too. Resolve only a
    # unique declared identity; never infer aliases from text or filenames.
    for identity, keys in declared_ids.items():
        if len(keys) == 1 and identity not in evidence:
            evidence[identity] = {'alias_of':keys[0], 'authority':'declared_runtime_evidence_identity'}
    return evidence


def scope_identity(args, result):
    return {'protocol':'x-editor.runtime-scopes/v1',
            'probe_binding_policy':'independent-probes/sample-frame-eof-v2',
            'listening_scope_rationale_visible':False,
            'runtime_evidence_fingerprint':canonical_evaluation_fingerprint(runtime_scope_evidence(result)),
            'semantic_provider':semantic_identity(args),
            'independent_speech_observer': (SpeechObservationConfig.model_validate(args.scope_speech_observer)
                .model_dump(mode='json') if getattr(args,'scope_speech_observer',None) else None),
            'answer_revision_attempts':getattr(args,'scope_answer_revision_attempts',0),
            'signal_measurements':getattr(args,'scope_signal_measurements',False),
            'audio_judgment_protocol':getattr(args,'audio_judgment_protocol','legacy'),
            'prompts':[SCOPE_PLAN_PROMPT,SCOPE_COVERAGE_PROMPT,LISTENING_PACKET_PROMPT,
                       DESCRIBE_PROMPT,CHECK_PROMPT,WINDOW_COORDINATE_PROMPT],
            'budgets':[getattr(args,'scope_max_rounds',2),getattr(args,'scope_max_observations',2),
                       getattr(args,'scope_max_probes',18)]}


def assess_runtime_regions(reviewer, scenario, result, artifact, output, invoke):
    assets = {a.asset_ref:{'path':a.path,'sha256':a.content_sha256,'role':a.role}
              for a in scenario.audio_assets}
    assets['terminal_candidate'] = {'path':artifact['path'],'sha256':artifact['sha256'],'role':'candidate'}
    context = public_context(scenario.model_dump(mode='json'), len(scenario.turns)-1)
    def semantic(prompt, payload, destination):
        from x_editor.revision.scope_preparation_cache import prepare_once
        cache = (getattr(reviewer, 'scope_preparation_cache', reviewer.cache)
                 if getattr(reviewer.args, 'reuse_scope_preparation', False) else None)
        def request():
            sent_prompt, sent_payload = prompt, payload
            compact = getattr(reviewer.args, 'compact_scope_preparation', False)
            if compact:
                from x_editor.evaluation.compact_scope_transport import prepare_request, expand_plan
                sent_prompt, sent_payload = prepare_request(prompt, payload)
            raw = invoke(destination,
                {'system_prompt':sent_prompt,'user_prompt':json.dumps(sent_payload,ensure_ascii=False)},
                reviewer.args, author=True)
            if compact and prompt == SCOPE_PLAN_PROMPT:
                # Raw provider response remains in the recorded call. Expand
                # exact references only, before all ordinary scope checks.
                expanded = expand_plan(raw)
                write_record(destination/'expanded_plan.json', expanded)
                return expanded
            return raw
        return prepare_once(cache=cache, prompt=prompt, payload=payload,
            args=reviewer.args, assets=assets, destination=destination,
            call=request)
    args = reviewer.args
    speech_config = getattr(args,'scope_speech_observer',None)
    assessment = review_runtime_scopes(
        design=reviewer.utility_design, public_context=context.model_dump(mode='json'),
        assets=assets, runtime_evidence=runtime_scope_evidence(result), output=output/'runtime_scopes',
        semantic_call=semantic, audio_call=reviewer.audio_call,
        max_rounds=getattr(args,'scope_max_rounds',2), max_observations=getattr(args,'scope_max_observations',2),
        max_probes=getattr(args,'scope_max_probes',18), concurrency=reviewer.audio_config.get('concurrency',1),
        speech_observer=ScopeSpeechObserver(speech_config) if speech_config else None,
        answer_revision_attempts=getattr(args,'scope_answer_revision_attempts',0),
        signal_measurements=getattr(args,'scope_signal_measurements',False),
        judgment_protocol=getattr(args,'audio_judgment_protocol','legacy'),
        configuration_fingerprint=canonical_evaluation_fingerprint(scope_identity(args, result)))
    write_record(output/'utility.json', assessment.model_dump(mode='json'))
    return assessment
