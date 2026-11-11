"""Execute validated Python commands through the common step verifier/handler."""
import ast
from types import SimpleNamespace

from x_editor.planner.command_program import inspect_commands


class CommandExecutionStopped(RuntimeError):
    pass


def _resolve(value, results):
    if isinstance(value, dict):
        if set(value) == {'from_step', 'from_output'}:
            return results[value['from_step']][value['from_output']]
        return {k: _resolve(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, results) for v in value]
    return value


async def execute_commands(plan, state, execute_one_step, sync_step=None):
    record = plan.metadata['command_program']
    original, calls, tree = inspect_commands(record['program'], record['catalog'])
    # Admission may attach evidence, but cannot change the authored program.
    for a, b in zip(original.steps, plan.steps, strict=True):
        for key, value in record.get('frozen_params', {}).get(a.id, {}).items():
            if key in a.params and a.params[key] != value:
                raise ValueError('command conflicts with a runtime-owned parameter')
            a.params[key] = value
        for field in ('id', 'operation', 'inputs', 'params', 'depends_on', 'span_timebase'):
            if getattr(a, field) != getattr(b, field):
                raise ValueError(f'admitted command changed {a.id}.{field}')
    if original.output_artifacts != plan.output_artifacts:
        raise ValueError('admitted command changed delivery')
    if state.get('suffix_resume_reused_step_ids'):
        raise ValueError('direct commands do not yet support prefix reuse')
    results, emitted = {}, []
    tools = SimpleNamespace()
    index = 0

    def function(operation):
        async def invoke(**arguments):
            nonlocal index
            expected = calls[index]
            arguments.setdefault('_timebase', 'observed_audio')
            if operation != expected['operation'] or arguments != _resolve(expected['arguments'], results):
                raise ValueError('runtime command differs from its statically admitted call')
            step = plan.steps[index]
            index += 1
            await execute_one_step(step, state)
            if sync_step:
                sync_step(step)
            if step.status != 'success':
                raise CommandExecutionStopped(step.error or f'{step.id} failed')
            output = dict(step.result or {})
            # The shared executor exposes this canonical output in step_outputs;
            # retain all additional named stems returned by the handler.
            path = state.get('step_outputs', {}).get(step.id)
            if isinstance(path, str):
                output.setdefault('output_audio_path', path)
            results[step.id] = output
            return output
        return invoke

    for operation in record['catalog']:
        namespace = tools
        parts = operation.split('.')
        for part in parts[:-1]:
            if not hasattr(namespace, part):
                setattr(namespace, part, SimpleNamespace())
            namespace = getattr(namespace, part)
        setattr(namespace, parts[-1], function(operation))

    def deliver(**kwargs):
        expected = original.output_artifacts[len(emitted)]
        value = results[expected.from_step][expected.from_output]
        if kwargs != dict(artifact_id=expected.artifact_id, value=value,
                          role=expected.role, primary=expected.primary):
            raise ValueError('runtime delivery differs from admission')
        emitted.append(kwargs)

    # Only the fully inspected AST is compiled. User code cannot access globals,
    # imports, arbitrary attributes, or builtins through this grammar.
    function_ast = ast.AsyncFunctionDef(name='run', args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg='tools'), ast.arg(arg='deliver')],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=tree.body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function_ast], type_ignores=[]))
    namespace = {'__builtins__': {}}
    exec(compile(module, '<x-editor-command>', 'exec'), namespace)
    audit = plan.metadata['command_execution'] = dict(
        backend='validated_python_functions', invoked=0, deliveries=0)
    try:
        await namespace['run'](tools, deliver)
    except CommandExecutionStopped as exc:
        audit['error'] = str(exc)
    finally:
        audit.update(invoked=index, deliveries=len(emitted))
        for step in plan.steps[index:]:
            step.status = 'skipped'
            step.error = 'Command stopped before this function call'
            if sync_step:
                sync_step(step)
