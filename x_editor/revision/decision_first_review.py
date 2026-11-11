"""Production review support extracted without changing function bodies."""

from dataclasses import dataclass


import json


from x_editor.revision.atomic_module_review import public_context


from x_editor.revision.refinement_checkpoints import latest_module_sources


from x_editor.evaluation.atomic_module_rubrics import bind_design, module_questions


from x_editor.evaluation.decision_review_gate import decision_review_gate


from x_editor.evaluation.iterative_runtime import write_record


from x_editor.evaluation.minimal_atomic_review import build_catalog


from x_editor.evaluation.refinement_diagnostics import executable_contract_question


from x_editor.evaluation.adviser_projection import absent_adviser


@dataclass
class DecisionReview:
    sources: dict
    prepared: dict
    catalogs: dict
    atomic: list
    errors: list
    gate: dict


def review_decisions(reviewer, *, scenario, result, directory, output):
    """Reasoner then Planner per turn; review errors cannot open the audio gate."""
    errors, prepared, catalogs, atomic = [], {}, {}, []
    try:
        sources, extraction_errors = latest_module_sources(scenario, result, directory)
        errors.extend(map(str, extraction_errors))
    except Exception as exc:
        sources = {}
        errors.append(f'decision checkpoints: {type(exc).__name__}: {exc}')
    if not reviewer.designs and hasattr(reviewer, 'module_design_provider'):
        try:
            reviewer.designs = reviewer.module_design_provider()
        except Exception as exc:
            errors.append(f'decision questions: {type(exc).__name__}: {exc}')
    expected = {(t.turn_index, 'planner') for t in result.turn_results} | set(sources)
    # Reasoner is optional in the Actor architecture. A present invocation must
    # be reviewed; a failed extraction is an error, never a silently passed memo.
    try:
        for path in (directory / 'turns').glob('*/execution/module_checkpoints.jsonl'):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                checkpoint = json.loads(line)
                if checkpoint['module'] == 'reasoner':
                    trace = checkpoint.get('output', {}).get('record', {}).get('invocation_trace')
                    if absent_adviser(trace):
                        continue
                    expected.add((checkpoint['context']['turn_index'], 'reasoner'))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f'expected decision checkpoints: {type(exc).__name__}: {exc}')
    qualified_designs = {}
    for key in sorted(sources, key=lambda k: (k[0], k[1] != 'reasoner')):
        turn, module = key
        try:
            context = public_context(scenario.model_dump(mode='json'), turn)
            design = bind_design(context, reviewer.designs[turn])
            if getattr(reviewer.args, 'decision_premise_review', False):
                from x_editor.revision.decision_premise_review import qualify_design
                if turn not in qualified_designs:
                    # One qualification per turn, shared by both module judges.
                    # Store exceptions too: a failed premise is not another try.
                    try:
                        qualified_designs[turn] = qualify_design(reviewer, context,
                            design, sources, turn=turn,
                            output=output/'premises'/f'{turn:04d}')
                    except Exception as exc:
                        qualified_designs[turn] = exc
                design = qualified_designs[turn]
                if isinstance(design, Exception):
                    raise design
            questions = module_questions(design, module)
            if module == 'planner':
                questions = (*questions, executable_contract_question())
            catalog = build_catalog(sources[key])
            catalogs[key] = catalog
            prepared[key] = design, questions, catalog, sources[key]
            atomic.append(reviewer.review_position(key, prepared[key], output))
        except Exception as exc:
            errors.append(f'{turn}/{module}: {type(exc).__name__}: {exc}')
    gate = decision_review_gate(atomic, expected_positions=expected, errors=errors,
                               minimum_grade=getattr(reviewer.args, 'decision_minimum_grade', 2))
    write_record(output / 'decision_gate.json', gate)
    return DecisionReview(sources, prepared, catalogs, atomic, errors, gate)
