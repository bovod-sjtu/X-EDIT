"""Descriptive construction workload; these counts are not value-function losses."""
import ast
import json


def count_references(value, key):
    if isinstance(value, dict):
        if key in value:
            return 1
        return sum(count_references(v, key) for v in value.values())
    if isinstance(value, list):
        return sum(count_references(v, key) for v in value)
    return 0


def measure_abstraction(authored, representation, lowering=None):
    text = authored.get('code', '') if representation == 'command' else json.dumps(authored, ensure_ascii=False)
    result = dict(program_characters=len(text), concrete_calls=None,
                  authored_function_units=None, compiler_added_calls=None,
                  expansion_ratio=None, compiler_added_call_fraction=None,
                  concrete_data_bindings=None, authored_data_bindings=None,
                  runtime_supplied_binding_fraction=None)
    if representation == 'planir':
        edits = authored.get('edits', [])
        effects = [e for edit in edits for e in edit.get('effects', [])]
        units = len(effects) + sum(e.get('kind') == 'concatenate' for e in edits)
        result.update(authored_function_units=units, local_edit_macros=sum(e.get('kind') == 'local_edit' for e in edits),
                      explicit_implementation_choices=sum(bool(e.get('implementation')) for e in effects),
                      authored_compositions=len(edits), authored_data_bindings=count_references(edits,'value_of'))
    elif representation == 'graph':
        graph = authored.get('graph', {})
        result.update(authored_function_units=len(graph.get('nodes', [])),
                      explicit_data_edges=len(graph.get('connections', [])),
                      explicit_order_edges=len(graph.get('ordering', [])),
                      authored_data_bindings=len(graph.get('connections', [])))
    elif representation == 'actions':
        result['authored_function_units'] = len(authored.get('actions', {}).get('calls', []))
        result['authored_data_bindings'] = count_references(authored.get('actions', {}),'result_of')
    elif representation == 'command':
        try:
            tree = ast.parse(text)
            result.update(authored_function_units=sum(isinstance(n, ast.Await) for n in ast.walk(tree)),
                          explicit_result_reads=sum(isinstance(n, ast.Subscript) for n in ast.walk(tree)),
                          authored_data_bindings=sum(isinstance(n,ast.Subscript) for s in tree.body
                                                    if isinstance(s,ast.Assign) for n in ast.walk(s)),
                          authored_syntax_nodes=sum(1 for _ in ast.walk(tree)))
        except SyntaxError:
            result['syntax_valid'] = False
    if lowering:
        concrete = lowering['concrete_calls']
        units = result['authored_function_units']
        if units:
            result.update(concrete_calls=concrete, compiler_added_calls=concrete-units,
                          expansion_ratio=concrete/units,
                          compiler_added_call_fraction=(concrete-units)/concrete if concrete else None)
        bindings=lowering.get('concrete_data_bindings')
        authored_bindings=result['authored_data_bindings']
        result['concrete_data_bindings']=bindings
        if bindings and authored_bindings is not None:
            result['runtime_supplied_binding_fraction']=(bindings-authored_bindings)/bindings
    return result
