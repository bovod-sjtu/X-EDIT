"""Project an authoritative delivery rejection without changing its raw receipt."""
from x_editor.runtime.workflow_session import WorkflowScenarioResult


def project_committed_attempt(stored, completion):
    result = WorkflowScenarioResult.model_validate(stored['result'])
    if stored.get('status', result.status) != completion['status']:
        raise ValueError('first result and completion disagree')
    if completion['status'] == 'artifact_missing' and result.status == 'completed':
        if completion.get('artifact_lineage') is not None:
            raise ValueError('rejected completion must not claim an accepted artifact')
        return WorkflowScenarioResult.model_validate({
            **result.model_dump(mode='json'), 'status': 'failed', 'failure_stage': 'delivery',
            'failure_codes': [*result.failure_codes, 'benchmark_artifact_not_accepted'],
            'terminal_error_type': 'ArtifactAcceptanceFailure',
            'terminal_error_message': 'The first benchmark attempt did not admit its produced artifact.',
            'final_output_path': None, 'final_state_projection': None,
        })
    if result.status != completion['status']:
        raise ValueError('first runtime result and committed status disagree')
    return result
