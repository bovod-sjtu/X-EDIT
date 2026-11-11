"""Optional coarse source observation, independent of the editing instruction.

This does not authorize edit boundaries, assign true speaker identities, or
route tools. The Reasoner must request finer evidence when the task needs it.
"""
from x_editor.runtime.workflow_orchestration import WorkflowPerceptionRequest

SCENE_INSTRUCTION = """Listen to this single audio asset and give a concise coarse scene
caption, independent of any editing goal. Describe audible speech/language and
transcript uncertainty, distinguishable speakers and uncertain voice attributes,
speaking manner (such as whispering, shouting or singing), the acoustic scene,
background music/noise/nonverbal events, overlap with speech, and audible room
reverberation or distortion. State unknown when a property cannot be established.
Do not infer demographic facts or speaker count from names mentioned in speech.
Distinguish 'not observed' from 'absent'. Do not invent precise event timestamps.
This is coarse evidence for choosing follow-up observations, not authorization to
edit an interval or a recommendation to select a tool. Preserve uncertainty and
bind every observation to this asset only."""


def scene_bootstrap_requests(case, profile):
    operation = 'perception.audio_fallback'
    route = profile.perception_route_policy(operation)
    return [WorkflowPerceptionRequest(
        request_id=f'semantic-bootstrap:scene:{index:04d}', scope='general',
        instruction=SCENE_INSTRUCTION, operation=operation,
        inputs={'input_audio_path': ref}, params=profile.perception_params(operation),
        expected_output='Coarse, asset-scoped audible scene observations with uncertainty.',
        audio_refs=[ref], requirements=['audio caption', 'explicit uncertainty for unavailable observations'],
        metadata={'framework_stage':'semantic_scene_observation',
                  'perception_route_policy':route.model_dump(mode='json')},
    ) for index,ref in enumerate(dict.fromkeys([*case.source_audio_refs,*case.reference_audio_refs]))]
