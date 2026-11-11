"""Opt-in action-space probes sharing PlanIR admission and execution.

Graph vertices expose concrete tool calls; edges bind output ports to JSON
argument paths. Sequential actions refer only to preceding call results. Neither
adapter selects operations, invents parameters, repairs failed outputs, or uses
benchmark identities. The production planning default is unchanged.
"""
from copy import deepcopy
from typing import Any, Literal
from pydantic import Field
from x_editor.core.schemas import AudioEditPlan, AudioEditStep, PlanOutputArtifact, StrictModel
from x_editor.runtime.execution_graph import compile_execution_graph


class Call(StrictModel):
    id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    description: str = ''
    expected_output: str | None = None
    span_timebase: Literal['observed_audio', 'step_input'] = 'observed_audio'
    required: bool = True


class Connection(StrictModel):
    producer: str
    output: str
    consumer: str
    argument_path: list[str | int] = Field(min_length=2)


class OrderEdge(StrictModel):
    before: str
    after: str


class GraphBody(StrictModel):
    nodes: list[Call] = Field(min_length=1)
    connections: list[Connection] = Field(default_factory=list)
    ordering: list[OrderEdge] = Field(default_factory=list)


class ActionBody(StrictModel):
    calls: list[Call] = Field(min_length=1)


class RepresentationPlan(StrictModel):
    request: str
    input_audio: str = Field(min_length=1)
    output_audio: str | None = None
    global_params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    missing_inputs: list[str] = Field(default_factory=list)
    output_artifacts: list[PlanOutputArtifact] = Field(default_factory=list)


class GraphPlan(RepresentationPlan):
    graph: GraphBody


class ActionsPlan(RepresentationPlan):
    actions: ActionBody


def _references(value):
    if isinstance(value, dict):
        if 'from_step' in value or 'from_output' in value:
            raise ValueError('Use representation-specific bindings, not embedded PlanIR references')
        for child in value.values():
            _references(child)
    elif isinstance(value, list):
        for child in value:
            _references(child)


def _bind(payload, path, reference):
    if path[0] not in ('inputs', 'params'):
        raise ValueError('Connection must target inputs or params')
    target = payload
    for segment in path[:-1]:
        target = target[segment]
    if target[path[-1]] is not None:
        raise ValueError('Connection must fill an explicit null argument, never overwrite a value')
    target[path[-1]] = reference


def _resolve_actions(value, previous, dependencies):
    if isinstance(value, dict):
        if 'result_of' in value:
            if set(value) != {'result_of', 'output'} or value['result_of'] not in previous:
                raise ValueError('Actions can reference only a declared preceding call')
            if not isinstance(value['output'], str) or not value['output']:
                raise ValueError('Result output name is required')
            dependencies.add(value['result_of'])
            return {'from_step': value['result_of'], 'from_output': value['output']}
        return {k: _resolve_actions(v, previous, dependencies) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_actions(v, previous, dependencies) for v in value]
    return value


def decode_plan(raw, representation: Literal['planir', 'graph', 'actions']):
    """Translate authored topology; run the same graph compiler in all arms."""
    if representation not in ('planir', 'graph', 'actions'):
        raise ValueError('Unknown action representation')
    if representation == 'planir':
        plan = AudioEditPlan.model_validate(raw)
    else:
        cls = GraphPlan if representation == 'graph' else ActionsPlan
        source = cls.model_validate(raw)
        calls = source.graph.nodes if representation == 'graph' else source.actions.calls
        ids = [c.id for c in calls]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate call IDs')
        values = {c.id: c.model_dump() for c in calls}
        dependencies = {i: set() for i in ids}
        for c in values.values():
            _references(c)
        if representation == 'graph':
            for edge in source.graph.ordering:
                if edge.before not in values or edge.after not in values:
                    raise ValueError('Unknown ordering vertex')
                dependencies[edge.after].add(edge.before)
            for edge in source.graph.connections:
                if edge.producer not in values or edge.consumer not in values:
                    raise ValueError('Unknown connection vertex')
                _bind(values[edge.consumer], edge.argument_path,
                      {'from_step': edge.producer, 'from_output': edge.output})
                dependencies[edge.consumer].add(edge.producer)
        else:
            for index, identifier in enumerate(ids):
                values[identifier] = _resolve_actions(values[identifier], set(ids[:index]), dependencies[identifier])
                # Keep schedule order separate from data edges: a serial call
                # can still expose an independently delivered terminal result.
        common = source.model_dump(exclude={'graph', 'actions'})
        plan = AudioEditPlan(**common, steps=[AudioEditStep(**values[i], depends_on=sorted(dependencies[i])) for i in ids])
    compile_execution_graph(plan, graph_id='representation_validation')
    return plan


REPRESENTATION_INSTRUCTIONS = {
 'graph': ('For an execute decision, replace only the plan representation with GraphPlan. '
           'graph.nodes are concrete tool calls. graph.connections explicitly bind producer/output '
           'to consumer/argument_path, e.g. ["inputs","input_audio"]. Put null in that argument. '
           'graph.ordering declares ordering-only edges. No hidden from_step references. '
           'Declare output_artifacts and preserve all ordinary evidence and tool-selection fields. '
           'The adapter never chooses tools or repairs a missing connection.'),
 'actions': ('For an execute decision, replace only the plan representation with ActionsPlan. '
             'actions.calls is the ordered list of concrete tool calls. Arguments may reference '
             'a preceding result using {"result_of":"call_id","output":"output_audio_path"}. '
             'No forward references or implicit current-output changes. Declare output_artifacts. '
             'Preserve all ordinary evidence and tool-selection fields.'),
}


class RepresentationClient:
    """Adapt only the provider's plan wire format, preserving raw call logs."""
    def __init__(self, client, representation):
        if representation not in ('planir', 'command', *REPRESENTATION_INSTRUCTIONS):
            raise ValueError('Unknown alternate representation')
        self.client, self.representation = client, representation
        self.raw_response = None

    def __getattr__(self, name):
        return getattr(self.client, name)

    @property
    def json_invocations(self):
        return self.client.json_invocations

    @json_invocations.setter
    def json_invocations(self, value):
        self.client.json_invocations = value

    def call_json(self, system, current, *, response_format, **kwargs):
        import json
        from x_editor.planner.decision_schema import bind_decision_payloads
        wire = bind_decision_payloads(response_format)
        if self.representation == 'planir':
            raw = self.client.call_json(system, current, response_format=wire, **kwargs)
            self.raw_response = deepcopy(raw)
            return raw
        schema = (GraphPlan if self.representation == 'graph' else ActionsPlan).model_json_schema()
        root = wire['json_schema']['schema']
        root.setdefault('$defs', {}).update(schema.pop('$defs', {}))
        count = 0
        def replace(node):
            nonlocal count
            if isinstance(node, dict):
                props = node.get('properties', {})
                if 'plan' in props:
                    props['plan'] = deepcopy(schema)
                    count += 1
                for k, v in node.items():
                    if k != '$defs': replace(v)
            elif isinstance(node, list):
                for v in node: replace(v)
        replace(root)
        if not count:
            raise ValueError('No plan slot found in shared decision schema')
        payload = json.loads(current)
        payload['action_representation'] = self.representation
        payload['representation_contract'] = REPRESENTATION_INSTRUCTIONS[self.representation]
        # Replace stale output-shape instructions rather than append two
        # contradictory output contracts. Public evidence is untouched.
        contract = payload.setdefault('planner_output_contract', {})
        body = ('graph:{nodes:[{id,operation,inputs,params}],connections,ordering}'
                if self.representation == 'graph' else
                'actions:{calls:[{id,operation,inputs,params}]}')
        from x_editor.planner.prompt_package import execute_decision_outline
        contract.setdefault('canonical_shapes', {})['execute'] = execute_decision_outline(
            '{request,input_audio,' + body +
            ',output_artifacts,global_params,metadata,missing_inputs}')
        contract['critical_rules'] = [rule for rule in contract.get('critical_rules', [])
            if not rule.startswith(('Plan step keys are exactly', 'Place tool arguments inside nested',
                                     'PlanIR input and parameter leaves'))]
        contract['representation_schema'] = {**schema, '$defs': root.get('$defs', {})}
        contract['common_schema_versions'] = {
            'execution_semantics': 'x-editor.planner-execution-semantics/v1',
            'tool_selection': 'x-editor.planner-tool-selection/v1'}
        raw = self.client.call_json(system+'\n'+REPRESENTATION_INSTRUCTIONS[self.representation],
                                    json.dumps(payload, ensure_ascii=False), response_format=wire, **kwargs)
        self.raw_response = deepcopy(raw)
        output = deepcopy(raw)
        turn = output.get('turn', output)
        if turn.get('plan') is not None:
            turn['plan'] = decode_plan(turn['plan'], self.representation).model_dump(mode='json')
        return output
