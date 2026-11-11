"""Versioned three-level program-construction experiment.

EditProgram alone undergoes high-level lowering. Concrete graphs and ordered
calls go directly to the existing common tool-program validation/executor.
The legacy AudioEditPlan carrier contains no high-level editing combinators.
"""
from copy import deepcopy
import json

from x_editor.planner.action_representation import (
    ActionsPlan, GraphPlan, RepresentationClient, decode_plan,
    REPRESENTATION_INSTRUCTIONS,
)
from x_editor.planner.edit_program import EditProgram, EditProgramCompiler, effect_catalog
from x_editor.planner.command_program import CommandProgram, COMMAND_INSTRUCTIONS, command_catalog, compile_commands


VERSION = 'hierarchical_v1'

EDIT_PROGRAM_INSTRUCTIONS = """For execute, plan is an EditProgram, NOT a table of tool calls.
Use edits of kind pipeline, local_edit, or concatenate and explicit deliver values.
pipeline declares a source and ordered effects. The compiler selects the unique
implementation of each effect (or the explicitly selected implementation when
several are available), maps semantic argument roles to tool parameters, threads
each output into the next effect, derives dependencies and resolves delivery.
Use effect_catalog for exact effect and semantic argument names. There is no
quality-based automatic tool selection. An ambiguous effect needs implementation.
argument_aliases lists exact alternative names for the same contract addresses;
use one name per address, never supply both an argument and its alias.
Do not write inputs, params, depends_on, connections, tool paths, or from_step.
Use {value_of: edit_or_effect_id, output: optional_named_port} for prior values.
Each edit and effect has a distinct ID. pipeline source=null is for generation
or explicit multi-input effects; otherwise the source is threaded automatically.
When a source is threaded, omit the effect's threaded_source_role argument.
In local_edit the threaded source is the cropped region, NOT the original file.
Effect spans are local to their threaded input (the crop starts at zero).
The compiler declares this coordinate system; source region bounds stay on source.
local_edit explicitly supplies source, region, effects,
preserve_outside=true, and duration_policy. Bounds must come from this source
version, never from a transcript end. The compiler crops, applies effects and
reassembles unselected audio. retain_generated_duration shifts the later timeline;
fit_region explicitly resizes the edited segment. Neither separates a background
source within the region: use an available separation/processing/remixing program
when component preservation requires it. Later coordinates refer to their bound
input version; do not reuse old timestamps after length-changing edits.
concatenate declares ordered sources, crossfade_ms and pause_duration_sec.
Tool capability limitations, evidence requirements and runtime safety still apply.
execution_semantics, if supplied, binds abstract edit/effect IDs. Argument paths
are source, region, sources, or arguments.<semantic_role>.
For example a volume effect e uses {plan_step_id:"e", operation:"relative_level_change",
target:"reduce level", argument_sources:[{argument_path:"arguments.relative_amplitude_change_db",
source_kind:"user_request"}], unresolved_requirements:[]}.
The containing pipeline p can separately bind {plan_step_id:"p",operation:"pipeline",
target:"source speech",argument_sources:[{argument_path:"source",source_kind:"user_request"}],
unresolved_requirements:[]}. Its audio_usage_binding uses plan_step_id=p and
argument_path=source. Never use inputs.input_audio or params.gain_db in these
abstract annotations. Local region evidence belongs to the local_edit ID.
The compiler only relocates supplied evidence; it cannot invent missing evidence.
Do not produce tool_selection: tool contracts and actual arguments are still
checked by the shared validator. All three study conditions omit this redundant
per-call comparison report. Decision summaries and evidence remain available.
"""


class AbstractionClient(RepresentationClient):
    def __init__(self, client, representation, capability_cards):
        super().__init__(client, representation)
        self.cards = capability_cards
        self.last_lowering = None

    def call_json(self, system, current, *, response_format, **kwargs):
        from x_editor.planner.decision_schema import bind_decision_payloads
        wire = bind_decision_payloads(response_format)
        cls = {'planir': EditProgram, 'graph': GraphPlan, 'actions': ActionsPlan,
               'command': CommandProgram}[self.representation]
        schema = cls.model_json_schema()
        root = wire['json_schema']['schema']
        root.setdefault('$defs', {}).update(schema.pop('$defs', {}))
        count = 0
        def replace(node):
            nonlocal count
            if isinstance(node, dict):
                props = node.get('properties', {})
                if 'plan' in props:
                    props['plan'] = deepcopy(schema)
                    props['tool_selection'] = {'type': 'null'}
                    count += 1
                for key, value in node.items():
                    if key != '$defs': replace(value)
            elif isinstance(node, list):
                for value in node: replace(value)
        replace(root)
        if not count:
            raise ValueError('No execute payload in shared decision schema')
        if self.representation == 'planir':
            paths = ['source', 'region', 'sources', 'crossfade_ms', 'pause_duration_sec']
            paths += sorted({f'arguments.{role}' for row in effect_catalog(self.cards) for role in row['arguments']})
            paths += sorted({f'arguments.{role}' for row in effect_catalog(self.cards) for role in row['argument_aliases']})
            def abstract_paths(node):
                if isinstance(node, dict):
                    props = node.get('properties', {})
                    if 'argument_path' in props:
                        props['argument_path'] = dict(type='string', enum=paths)
                    for child in node.values(): abstract_paths(child)
                elif isinstance(node, list):
                    for child in node: abstract_paths(child)
            abstract_paths(root)
        payload = json.loads(current)
        instruction = (COMMAND_INSTRUCTIONS if self.representation == 'command' else
                       EDIT_PROGRAM_INSTRUCTIONS if self.representation == 'planir' else
                       REPRESENTATION_INSTRUCTIONS[self.representation].replace(' and tool-selection', '') +
                       '\nDo not produce tool_selection. Evidence, asset use, tool contracts and safety checks remain required.')
        payload['action_abstraction_version'] = VERSION
        payload['action_representation'] = self.representation
        # Same public effect/implementation information in all conditions.
        payload['effect_catalog'] = effect_catalog(self.cards)
        if self.representation == 'command':
            payload['command_api'] = command_catalog(self.cards)
        contract = payload.setdefault('planner_output_contract', {})
        contract['representation_contract'] = instruction
        contract['representation_schema'] = {**schema, '$defs': root.get('$defs', {})}
        from x_editor.planner.prompt_package import execute_decision_outline
        shape = {'planir': '{request,input_audio,edits,deliver}',
                 'graph': '{request,input_audio,graph:{nodes,connections,ordering},output_artifacts}',
                 'actions': '{request,input_audio,actions:{calls},output_artifacts}',
                 'command': '{request,input_audio,code}'}[self.representation]
        contract.setdefault('canonical_shapes', {})['execute'] = execute_decision_outline(shape).replace(
            ', tool_selection?:{steps:[{plan_step_id,selected_operation,candidates,selection_summary}]}', '')
        contract['critical_rules'] = [rule for rule in contract.get('critical_rules', [])
            if not any(text in rule for text in ('Plan step keys are exactly', 'Place tool arguments inside nested',
                 'PlanIR input and parameter leaves', 'tool_selection'))]
        system = system.replace('PlanIR', 'the declared program representation')
        raw = self.client.call_json(system+'\n'+instruction, json.dumps(payload, ensure_ascii=False),
                                    response_format=wire, **kwargs)
        self.raw_response = deepcopy(raw)
        self.last_lowering = None
        result = deepcopy(raw)
        turn = result.get('turn', result)
        if turn.get('plan') is None:
            return result
        if turn.get('tool_selection') is not None:
            raise ValueError('hierarchical_v1 uses common tool validation, not a tool_selection report')
        if self.representation == 'planir':
            compiler = EditProgramCompiler(self.cards)
            concrete = compiler.compile(turn['plan'])
            turn['execution_semantics'] = compiler.project_semantics(turn.get('execution_semantics'))
            mapping = compiler.mapping
        elif self.representation == 'command':
            concrete = compile_commands(turn['plan'], self.cards)
            mapping = [dict(step_id=s.id, origin=s.id, operation=s.operation) for s in concrete.steps]
        else:
            # These decoders only bind explicit concrete calls. They cannot
            # invoke EditProgramCompiler, select operations, or expand scope.
            concrete = decode_plan(turn['plan'], self.representation)
            mapping = [dict(step_id=s.id, origin=s.id, operation=s.operation) for s in concrete.steps]
        from x_editor.planner.abstraction_metrics import count_references
        self.last_lowering = dict(version=VERSION, representation=self.representation,
            authored_plan=deepcopy(turn['plan']), mappings=mapping,
            concrete_data_bindings=sum(count_references([s.inputs,s.params], 'from_step') for s in concrete.steps),
            concrete_calls=len(concrete.steps), high_level_compiler_used=self.representation == 'planir')
        concrete.metadata['action_abstraction'] = deepcopy(self.last_lowering)
        turn['plan'] = concrete.model_dump(mode='json')
        return result
