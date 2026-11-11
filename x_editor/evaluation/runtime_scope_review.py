"""Bounded scope planning, additional observation, actual crops and coverage.

Injected recorded calls keep this independent of service configuration. It never
executes an edit, sees private annotations, or substitutes a tool's success for
listening evidence. A coverage audit is a model judgment, not an oracle.
"""

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import time

import soundfile as sf

from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint
from x_editor.evaluation.iterative_runtime import sha256, write_record
from x_editor.evaluation.refinement_contracts import ObservationCheck, UtilityAnswer, UtilityAssessment
from x_editor.evaluation.refinement_observation import recheck_observation
from x_editor.evaluation.regional_listening import (
    ListeningProbe, ListeningWindow, assess_listening_packet, prepare_listening_packet,
)
from x_editor.evaluation.runtime_scope_contracts import (
    RuntimeScopeCoverage, RuntimeScopePlan, SCOPE_COVERAGE_PROMPT, SCOPE_PLAN_PROMPT,
)


def exact_ids(rows, expected):
    ids = [r.criterion_id for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("scope records must cover each frozen criterion exactly once")


def proposed_coverage(plan, inventory):
    """Expose unheard spans; the semantic audit decides which ones matter.

This arithmetic does not know the correct edit region, and deliberately never
labels a gap as authorized or a union as complete semantic coverage.
"""
    rows = []
    for scope in plan.criteria:
        for asset in inventory:
            ranges = sorted((w.start_sec, w.end_sec) for p in scope.probes
                            for w in (p.candidate, *p.references) if w.audio_ref == asset['audio_ref'])
            merged = []
            for start, end in ranges:
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start,end])
            gaps, cursor = [], 0.
            for start, end in merged:
                if start > cursor:
                    gaps.append([cursor,start])
                cursor = max(cursor,end)
            if cursor < asset['duration_sec']:
                gaps.append([cursor,asset['duration_sec']])
            rows.append({'criterion_id':scope.criterion_id,'audio_ref':asset['audio_ref'],
                         'proposed_union_sec':merged,'not_proposed_sec':gaps})
    return rows


def bind_window(window, assets, evidence):
    asset = assets.get(window.audio_ref)
    if not asset or set(window.evidence_ids) - set(evidence):
        raise ValueError("unknown audio version or grounding evidence")
    info = sf.info(asset['path'])
    # Decimal seconds can round an EOF timestamp slightly past the last sample
    # (e.g. 6.880145833 -> 6.88015). Bind that sub-sample excess to EOF;
    # requests extending by more than one sample still lack valid coverage.
    if window.end_sec * info.samplerate > info.frames + 1:
        raise ValueError("scope exceeds the actual audio duration")
    return ListeningWindow(
        audio_ref=window.audio_ref, sha256=asset['sha256'],
        start_frame=math.floor(window.start_sec * info.samplerate),
        end_frame=min(info.frames, math.ceil(window.end_sec * info.samplerate)),
        boundary_frame=None if window.boundary_sec is None else round(window.boundary_sec * info.samplerate),
        basis_evidence_ids=window.evidence_ids,
    )


def bind_probe(draft, criterion, index, assets, evidence):
    return ListeningProbe(
        probe_id=f"{criterion.criterion_id}.{index}", parent_criterion_id=criterion.criterion_id,
        focus=criterion.region.focus, question=draft.question,
        scope_status='ready_for_listening', scope_explanation=draft.explanation,
        candidate=bind_window(draft.candidate, assets, evidence),
        references=tuple(bind_window(w, assets, evidence) for w in draft.references),
    )


def aggregate_criterion(criterion, scope, coverage, local, *, design_gaps=()):
    """One supported counterexample can fail; partial positive listening cannot pass."""
    supported = set(coverage.supported_probe_indices) if coverage else set()
    values = [local[i][0] for i in supported if i in local]
    complete = bool(coverage and coverage.coverage == 'complete' and not design_gaps
                    and not coverage.uncovered_conditions and not scope.missing_evidence
                    and supported == set(range(len(scope.probes))) and supported
                    and len(values) == len(supported) and all(v.grade is not None for v in values))
    failures = [v.grade for v in values if v.grade is not None and v.grade < 2]
    grade = min(failures) if failures else min(v.grade for v in values) if complete else None
    finding = ' | '.join(v.finding for v in values)
    if not complete:
        gaps = [*design_gaps, *scope.missing_evidence,
                *(coverage.uncovered_conditions if coverage else ('No valid coverage audit.',))]
        finding = 'Coverage unresolved: ' + '; '.join(gaps or ['not all supported windows were judged']) + '. ' + finding
    return UtilityAnswer(question_id=criterion.criterion_id, grade=grade,
                         finding=(finding or 'No valid local observation.')[:1400],
                         evidence_ids=tuple(dict.fromkeys(e for v in values for e in v.evidence_ids)) or ('scope_review',))


def review_runtime_scopes(*, design, public_context, assets, runtime_evidence, output,
                          semantic_call, audio_call, max_rounds=2, max_observations=2,
                          max_probes=18, concurrency=2, configuration_fingerprint=None,
                          speech_observer=None, answer_revision_attempts=0, signal_measurements=False,
                          judgment_protocol='legacy'):
    """Plan -> bounded observation -> replan -> coverage -> listen -> assessment.

max_rounds includes the first plan; each additional observation makes two calls
(blind description then claim check). Unknown listening is passed to the existing
Guider, rather than starting an unbounded internal repair/reauthoring loop.
"""
    if type(max_rounds) is not int or not 1 <= max_rounds <= 3:
        raise ValueError('scope planning must use one to three rounds')
    if type(max_observations) is not int or not 0 <= max_observations <= 6:
        raise ValueError('scope observation budget must be zero to six')
    if type(max_probes) is not int or not 1 <= max_probes <= 48:
        raise ValueError('scope listening budget must be one to 48 probes')
    started = time.monotonic()
    output = Path(output)
    criteria = {c.criterion_id: c for c in design.criteria}
    inventory = []
    for ref, asset in assets.items():
        if sha256(asset['path']) != asset['sha256']:
            raise ValueError('audio changed before scope planning')
        info = sf.info(asset['path'])
        inventory.append({'audio_ref':ref, 'role':asset['role'], 'sha256':asset['sha256'],
                          'duration_sec':info.frames/info.samplerate,
                          'sample_rate':info.samplerate, 'frames':info.frames})
    candidates = [a for a in inventory if a['role'] == 'candidate']
    if len(candidates) != 1:
        raise ValueError('scope review needs one complete terminal candidate')
    evidence = {'public_request':public_context, 'audio_geometry':inventory, **runtime_evidence}
    methods = {'listen': '先盲听，再核查请求中的事件声称；最多两次多模调用。'}
    if speech_observer is not None:
        methods['speech_grounding'] = speech_observer.description
    write_record(output / 'input.json', {'design':design.model_dump(mode='json'),
                 'public_context':public_context, 'audio_inventory':inventory,
                 'evidence_catalog':evidence, 'private_scope_visible':False,
                 'available_observation_methods':methods})
    errors, active_errors, history, observed_keys = [], [], [], set()
    invalid_probes = {}
    observation_count = 0
    plan, coverage, previous, pending_requests = None, None, None, []
    for round_index in range(max_rounds):
        active_errors, invalid_probes = [], {}
        root = output / f'round_{round_index:02d}'
        payload = {'design':design.model_dump(mode='json'), 'public_context':public_context,
                   'audio_inventory':inventory, 'evidence_catalog':evidence,
                   'max_total_probes':max_probes, 'observations_remaining':max_observations-observation_count,
                   'available_observation_methods':methods,
                   'available_signal_measurements':['aligned_pcm_comparison'] if signal_measurements else [],
                   'previous_review':previous}
        try:
            raw = semantic_call(SCOPE_PLAN_PROMPT, {**payload, 'schema':RuntimeScopePlan.model_json_schema()}, root/'plan_call')
            write_record(root/'plan_original.json', raw)
            plan = RuntimeScopePlan.model_validate(raw)
            exact_ids(plan.criteria, criteria)
            if sum(len(s.probes) for s in plan.criteria) > max_probes:
                raise ValueError('scope plan exceeds the total listening budget')
            # Reject invalid coordinates/roles while a replan is still possible,
            # before paying for a coverage audit and listening. This validates
            # the model's proposal, never substitutes a different window.
            for scope in plan.criteria:
                for index, draft in enumerate(scope.probes):
                    try:
                        bind_probe(draft, criteria[scope.criterion_id], index, assets, evidence)
                        if (assets[draft.candidate.audio_ref]['role'] != 'candidate'
                                or any(assets[w.audio_ref]['role'] == 'candidate' for w in draft.references)):
                            raise ValueError('candidate window must use the terminal candidate; '
                                             'references must use source/reference assets')
                        if draft.signal_measurement != 'none' and not signal_measurements:
                            raise ValueError('requested signal measurement is unavailable in this protocol')
                    except ValueError as exc:
                        message = f'{scope.criterion_id}/{index}: {exc}'
                        invalid_probes[(scope.criterion_id, index)] = message
                        active_errors.append(message)
                        errors.append(f'scope round {round_index}: {message}')
            binding_errors = [{'criterion_id':key[0], 'probe_index':key[1], 'error':value}
                              for key, value in invalid_probes.items()]
            write_record(root/'probe_binding_errors.json', binding_errors)
            payload['ineligible_probes'] = binding_errors
            coverage = None
            # Obtain already-requested evidence before auditing a provisional
            # plan. Every plan that reaches listening still needs the audit.
            obtain_first = bool(round_index+1 < max_rounds and (invalid_probes or
                                (plan.observation_requests and observation_count < max_observations)))
            if not obtain_first:
                raw = semantic_call(SCOPE_COVERAGE_PROMPT,
                    {**payload, 'plan':plan.model_dump(mode='json'),
                     'coordinate_coverage':proposed_coverage(plan, inventory),
                     'schema':RuntimeScopeCoverage.model_json_schema()}, root/'coverage_call')
                write_record(root/'coverage_original.json', raw)
                coverage = RuntimeScopeCoverage.model_validate(raw)
                exact_ids(coverage.criteria, criteria)
                scopes = {s.criterion_id:s for s in plan.criteria}
                for c in coverage.criteria:
                    indices = c.supported_probe_indices
                    if len(indices) != len(set(indices)) or any(i < 0 or i >= len(scopes[c.criterion_id].probes) for i in indices):
                        raise ValueError('coverage cites a nonexistent or duplicate probe')
            previous = {'plan':plan.model_dump(mode='json'), 'coverage':coverage.model_dump(mode='json') if coverage else None}
            if binding_errors:
                previous.update(ineligible_probes=binding_errors,
                                invalid_response='; '.join(active_errors))
            history.append(previous)
            requests = [*plan.observation_requests, *(coverage.observation_requests if coverage else ())]
            for request in requests:
                if (set(request.criterion_ids) - set(criteria) or set(request.evidence_ids) - set(evidence)
                        or request.audio_ref not in assets or request.method not in methods):
                    raise ValueError('scope observation must cite available criteria, evidence and audio')
            pending_requests = requests
        except Exception as exc:
            error = f'scope round {round_index}: {type(exc).__name__}: {exc}'
            errors.append(error)
            active_errors.append(error)
            write_record(root/'error.json', {'error':error})
            previous = {'invalid_response':str(exc)}
            # A partially parsed/invalid audit cannot authorize a quality grade.
            plan, coverage = None, None
            continue
        if round_index + 1 == max_rounds:
            break  # no observations that cannot be incorporated in a new plan
        new_observation = False
        for request in requests:
            key = canonical_evaluation_fingerprint(request)
            if key in observed_keys or observation_count >= max_observations:
                continue
            observed_keys.add(key)
            observation_count += 1
            destination = output / 'additional_observations' / f'{observation_count:02d}'
            try:
                if set(request.criterion_ids) - set(criteria) or set(request.evidence_ids) - set(evidence):
                    raise ValueError('additional observation cites an unknown criterion or evidence')
                asset = assets[request.audio_ref]
                if request.method == 'speech_grounding':
                    receipt = speech_observer(request, asset, destination)
                else:
                    info = sf.info(asset['path'])
                    check = ObservationCheck(turn_index=0, audio_ref=request.audio_ref, audio_sha256=asset['sha256'],
                        start_sec=request.start_sec, end_sec=request.end_sec or info.frames/info.samplerate,
                        event_claim=request.event_claim, evidence_ids=request.evidence_ids)
                    receipt = recheck_observation(check, source=Path(asset['path']), output=destination, call=audio_call)
                evidence[f'additional_observation_{observation_count}'] = receipt
            except Exception as exc:
                error = f'additional observation {observation_count}: {type(exc).__name__}: {exc}'
                errors.append(error)
                write_record(destination/'failure.json', {'error':error})
                evidence[f'additional_observation_{observation_count}'] = {'error':error, 'request':request.model_dump(mode='json')}
            new_observation = True
        if not new_observation:
            needs_scope_revision = bool(invalid_probes or (coverage and any(
                c.coverage != 'complete' or c.uncovered_conditions for c in coverage.criteria)))
            if not needs_scope_revision:
                break
    local_results, observations = {}, []
    scopes = {s.criterion_id:s for s in plan.criteria} if plan else {}
    for identity, scope in list(scopes.items()):
        pending = [r.event_claim for r in pending_requests if identity in r.criterion_ids]
        invalid = [message for (key, _), message in invalid_probes.items() if key == identity]
        if pending or invalid:
            scopes[identity] = scope.model_copy(update={'missing_evidence':
                (*scope.missing_evidence, *invalid, *('Requested observation remains unresolved: '+p for p in pending))})
    audits = {c.criterion_id:c for c in coverage.criteria} if coverage else {}
    jobs = [(criteria[identity], i, scope.probes[i]) for identity, scope in scopes.items()
            for i in (audits[identity].supported_probe_indices if identity in audits else ())
            if (identity, i) not in invalid_probes]

    def listen(job):
        criterion, index, draft = job
        root = output/'listening'/canonical_evaluation_fingerprint(criterion.criterion_id)[:20]/str(index)
        try:
            probe = bind_probe(draft, criterion, index, assets, evidence)
            packet, paths = prepare_listening_packet(probe, assets, evidence, root/'packet')
            if draft.signal_measurement != 'none':
                if not signal_measurements:
                    raise ValueError('requested signal measurement is unavailable in this protocol')
                from x_editor.evaluation.listening_signal_facts import measure_listening_comparisons
                packet['signal_facts'] = measure_listening_comparisons(packet, paths)
                write_record(root/'packet/signal_facts.json', packet['signal_facts'])
            answer, observation = assess_listening_packet(probe, packet, paths, audio_call, root/'review',
                                                          include_scope_explanation=False,
                                                          revision_attempts=answer_revision_attempts,
                                                          judgment_protocol=judgment_protocol, semantic_call=semantic_call)
            return criterion.criterion_id, index, answer, observation, None
        except Exception as exc:
            error = f'listening {criterion.criterion_id}/{index}: {type(exc).__name__}: {exc}'
            write_record(root/'failure.json', {'error':error})
            return criterion.criterion_id, index, None, None, error

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for identity, index, answer, observation, error in pool.map(listen, jobs):
            if error:
                errors.append(error)
                active_errors.append(error)
            else:
                local_results.setdefault(identity, {})[index] = (answer, observation)
                observations.append(observation)
    answers = []
    for identity, criterion in criteria.items():
        if identity not in scopes or identity not in audits:
            answers.append(UtilityAnswer(question_id=identity, grade=None,
                           finding='No valid scope plan and coverage review; no positive quality conclusion.', evidence_ids=('scope_review',)))
        else:
            answers.append(aggregate_criterion(criterion, scopes[identity], audits[identity],
                local_results.get(identity, {}), design_gaps=coverage.missing_design_requirements))
    receipt = {'protocol':'x-editor.runtime-scopes/v1', 'private_scope_visible':False,
               'runtime_evidence_fingerprint':canonical_evaluation_fingerprint(runtime_evidence),
               'configuration_fingerprint':configuration_fingerprint,
               'coverage_is_model_judgment':True, 'rounds':len(history),
               'additional_observations':observation_count, 'attempted_listening_probes':len(jobs),
               'available_observation_methods':list(methods),
               'signal_measurements_enabled':signal_measurements,
               'probe_binding_policy':'independent-probes/v1',
               'error_history':errors,
               'new_observations':{f'additional_observation_{i}':evidence[f'additional_observation_{i}']
                                   for i in range(1,observation_count+1)},
               'completed_listening_probes':len(observations),
               'coverage':coverage.model_dump(mode='json') if coverage else None,
               'plan':plan.model_dump(mode='json') if plan else None,
               'elapsed_sec':time.monotonic()-started}
    assessment = UtilityAssessment(design_fingerprint=canonical_evaluation_fingerprint(design),
        artifact_sha256=candidates[0]['sha256'], answers=tuple(answers), errors=tuple(active_errors),
        assessor_role='online_selection_proxy', regional_observations=tuple(observations), scope_review=receipt)
    write_record(output/'assessment.json', assessment.model_dump(mode='json'))
    return assessment
