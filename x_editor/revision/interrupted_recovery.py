"""Production review support extracted without changing function bodies."""

from copy import deepcopy


import json


from pathlib import Path


from x_editor.core.future_attempt_guidance import FutureAttemptGuidance


from x_editor.evaluation.iterative_runtime import delivered_artifact, sha256


from x_editor.evaluation.refinement_contracts import FeedbackDraft, RefinementPolicy, utility_bounds


from x_editor.runtime.workflow_session import WorkflowScenarioResult


def read(path):
    return json.loads(Path(path).read_text())


def load_prefix(path, scenario, policy):
    state=read(path)
    recorded_policy=RefinementPolicy.model_validate(state['policy'])
    transition=state.get('efficiency_policy_transition')
    if transition:
        if (transition.get('changed_fields') != ['selection_assessment_reuse']
                or RefinementPolicy.model_validate(transition['from']) != recorded_policy
                or RefinementPolicy.model_validate(transition['to']) != policy
                or recorded_policy.model_copy(update={'selection_assessment_reuse':policy.selection_assessment_reuse}) != policy
                or state.get('terminal_status') not in {'Stopped','Failed'}):
            raise ValueError('invalid efficiency policy transition')
        recorded_policy=policy
    if (state.get('protocol')!='x-editor.interrupted-recovery-prefix/v1'
            or state['scenario_fingerprint']!=scenario.fingerprint
            or recorded_policy!=policy
            or state['next_attempt']>policy.max_retries+1
            or [r['attempt_index'] for r in state['attempts']]!=list(range(state['next_attempt']))):
        raise ValueError('interrupted recovery policy, identity or budget differs')
    for name,digest in state['preserved_files'].items():
        if sha256(name)!=digest:raise ValueError('interrupted evidence changed')
    outputs={}
    projections=[]
    from x_editor.evaluation.recorded_result_compatibility import read_recorded_result
    for index,item in state['results'].items():
        if sha256(item['path'])!=item['sha256']:raise ValueError('saved result changed')
        outputs[int(index)], changes = read_recorded_result(item['result'], WorkflowScenarioResult)
        projections.extend({'attempt_index':int(index), **change} for change in changes)
    for row in state['attempts']:
        artifact=row.get('artifact')
        if artifact and sha256(artifact['path'])!=artifact['sha256']:
            raise ValueError('saved audio changed')
    guidance={int(k):FutureAttemptGuidance.model_validate(v) for k,v in state['guidance'].items()}
    state=deepcopy(state)
    if projections:
        state['reader_compatibility_projections']=projections
    return state,outputs,guidance
