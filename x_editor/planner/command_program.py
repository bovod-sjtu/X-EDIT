"""Executable, straight-line Python commands over the public operation API.

Static inspection produces an admission record, not a high-level edit plan.
The runtime executes the *authored Python* and its actual function calls, using
the ordinary per-operation executor. No tool selection, crop, reconstruction,
implicit output threading, imports, loops, or arbitrary Python are supplied.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import keyword
import math
import re
from typing import Any

from pydantic import Field

from x_editor.core.schemas import AudioEditPlan, AudioEditStep, PlanOutputArtifact, StrictModel
from x_editor.runtime.execution_graph import compile_execution_graph


class CommandProgram(StrictModel):
    request: str
    input_audio: str = Field(min_length=1)
    code: str = Field(min_length=1, max_length=100000)


def command_catalog(cards):
    """Actual keyword API: parameter names and ownership come from tool cards."""
    catalog = {}
    for operation, card in sorted(cards.items()):
        if isinstance(card, dict):
            from x_editor.capabilities.schemas import OperationCapabilityCard
            card = OperationCapabilityCard.model_validate(card)
        if any(not p.isidentifier() or p.startswith('_') or keyword.iskeyword(p)
               for p in operation.split('.')):
            raise ValueError(f'operation has no Python API name: {operation}')
        paths = sorted({'.'.join(p.path.split('.')[:2])
                        for p in card.parameter_contract.planner_selectable
                        if p.path.startswith(('inputs.', 'params.'))})
        arguments = {}
        for path in paths:
            leaf = path.split('.')[1]
            name = leaf if sum(p.split('.')[1] == leaf for p in paths) == 1 else path.replace('.', '_')
            if not name.isidentifier() or keyword.iskeyword(name) or name.startswith('_'):
                raise ValueError(f'parameter has no Python API name: {path}')
            arguments[name] = path
        catalog[operation] = arguments
    return catalog


def _operation(call):
    parts = []
    node = call.func
    while isinstance(node, ast.Attribute):
        if node.attr.startswith('_'):
            raise ValueError('private attributes are not command functions')
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id != 'tools' or not parts:
        raise ValueError('call a registered tools.<operation> function')
    return '.'.join(reversed(parts))


def _value(node, previous):
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise ValueError('nonfinite argument')
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        value = node.operand.value
        if type(value) in (int, float):
            return -value
    if isinstance(node, ast.List):
        return [_value(x, previous) for x in node.elts]
    if isinstance(node, ast.Dict):
        keys = [_value(k, previous) for k in node.keys if k is not None]
        if len(keys) != len(node.keys) or any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
            raise ValueError('dictionary keys must be unique literal strings')
        return dict(zip(keys, [_value(v, previous) for v in node.values]))
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id in previous and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str) and node.slice.value):
        return dict(from_step=node.value.id, from_output=node.slice.value)
    raise ValueError('arguments must be literals or named outputs of preceding calls')


def _kwargs(call, previous):
    if call.args or any(k.arg is None for k in call.keywords):
        raise ValueError('commands require named arguments; no *args or **kwargs')
    names = [k.arg for k in call.keywords]
    if len(names) != len(set(names)):
        raise ValueError('duplicate keyword argument')
    return {k.arg: _value(k.value, previous) for k in call.keywords}


def _dependencies(value):
    if isinstance(value, dict):
        if 'from_step' in value:
            if set(value) != {'from_step', 'from_output'}:
                raise ValueError('malformed result binding')
            return {value['from_step']}
        return set().union(*(_dependencies(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_dependencies(v) for v in value))
    return set()


def inspect_commands(raw, catalog):
    """Reject unsupported syntax before any function is called."""
    program = CommandProgram.model_validate(raw)
    tree = ast.parse(program.code, mode='exec')
    steps, outputs, previous = [], [], set()
    calls = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            identifier = statement.targets[0].id
            if outputs or identifier in previous or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', identifier) or identifier in ('tools', 'deliver'):
                raise ValueError('unique call variables must precede delivery')
            if not isinstance(statement.value, ast.Await) or not isinstance(statement.value.value, ast.Call):
                raise ValueError('assign the result of await tools.<operation>(...)')
            call = statement.value.value
            operation = _operation(call)
            if operation not in catalog:
                raise ValueError(f'unknown command function: {operation}')
            arguments = _kwargs(call, previous)
            timebase = arguments.pop('_timebase', 'observed_audio')
            if timebase not in ('observed_audio', 'step_input'):
                raise ValueError('invalid command coordinate system')
            inputs, params = {}, {}
            for key, value in arguments.items():
                if key not in catalog[operation]:
                    raise ValueError(f'unknown argument {operation}.{key}')
                owner, name = catalog[operation][key].split('.')
                (inputs if owner == 'inputs' else params)[name] = value
            # Only real Python references can create bindings, never a disguised
            # JSON IR supplied as a literal dictionary.
            literal_refs = [n for n in ast.walk(call) if isinstance(n, ast.Dict)
                            and any(isinstance(k, ast.Constant) and k.value in ('from_step', 'result_of') for k in n.keys)]
            if literal_refs:
                raise ValueError('use a Python result variable, not an IR reference dictionary')
            steps.append(AudioEditStep(id=identifier, operation=operation, inputs=inputs, params=params,
                                       span_timebase=timebase, depends_on=sorted(_dependencies([inputs, params]))))
            calls.append(dict(id=identifier, operation=operation,
                              arguments={**arguments, '_timebase': timebase}))
            previous.add(identifier)
        elif (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
              and isinstance(statement.value.func, ast.Name) and statement.value.func.id == 'deliver'):
            values = _kwargs(statement.value, previous)
            if set(values) != {'artifact_id', 'value', 'role', 'primary'}:
                raise ValueError('deliver requires artifact_id, value, role, primary')
            ref = values.pop('value')
            if not isinstance(ref, dict) or set(ref) != {'from_step', 'from_output'}:
                raise ValueError('deliver an actual named function output')
            outputs.append(PlanOutputArtifact(**values, **ref))
        else:
            raise ValueError('only awaited tool assignments and final deliver calls are supported')
    if not steps or not outputs:
        raise ValueError('commands require at least one function call and explicit delivery')
    plan = AudioEditPlan(request=program.request, input_audio=program.input_audio,
                         steps=steps, output_artifacts=outputs)
    compile_execution_graph(plan, graph_id='command_admission')
    return plan, calls, tree


def compile_commands(raw, cards):
    catalog = command_catalog(cards)
    plan, calls, _ = inspect_commands(raw, catalog)
    # Runtime gets just the APIs used by this program, not an invented function
    # implementation. Calls still dispatch the shared registered operation.
    plan.metadata['command_program'] = dict(
        version='python_commands_v1', program=deepcopy(raw),
        catalog={s.operation: catalog[s.operation] for s in plan.steps},
        source_sha256=hashlib.sha256(raw['code'].encode()).hexdigest(), calls=calls)
    return plan


COMMAND_INSTRUCTIONS = """For execute, plan is {request,input_audio,code}.
code is executable straight-line async Python, not a JSON IR or shell script.
Separate Python statements with semicolons or actual newlines (JSON-escaped).
Whitespace alone cannot separate an assignment and the next function call.
Example code: v = await tools.artifact_ops.apply_gain(input_audio="aud:source", gain_db=-6); deliver(artifact_id="final", value=v["output_audio_path"], role="main", primary=True)
Use actual functions from command_api: v = await tools.<operation>(named=value).
Read prior outputs as v["output_audio_path"] (or a named stem from its contract).
Use Python True/False/None, literal dictionaries/lists, and named arguments.
No imports, loops, branches, expressions, invented functions, or filesystem access.
Every concrete tool call, crop, source binding and reconstruction is your choice.
No implicit current output, automatic local edit expansion or tool selection.
For spans local to a previous crop, pass _timebase="step_input"; otherwise it
defaults to "observed_audio". This declares coordinates and does not create evidence.
Finish with deliver(artifact_id="final",value=v["output_audio_path"],role="main",primary=True).
Multiple explicit deliver calls are allowed. IDs in execution_semantics are Python
variable names. Evidence argument paths still use the API's inputs/params ownership
shown in command_api, so all representations have identical admission requirements.
Do not produce tool_selection. Do not repeat a JSON list of actions in code.
"""
