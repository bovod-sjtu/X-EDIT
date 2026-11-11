"""Production review support extracted without changing function bodies."""

from pathlib import Path


from x_editor.revision.progressive_recovery_inputs import shared_design


from x_editor.evaluation.refinement_contracts import parse_utility_design


from x_editor.evaluation.base_terminal_design import base_terminal_design


def deferred_terminal_provider(scenario, spec, args, audio_config):
    registry = spec['deferred_terminal_designs']

    def resolve():
        record = shared_design(scenario, Path(registry['root']), args, audio_config,
                               registry['identity'], stage='terminal')
        if record is None:
            return base_terminal_design() if spec.get('base_terminal_fallback', False) else None
        if record['qualification_status'] != 'qualified':
            if spec.get('base_terminal_fallback', False):
                # The failed draft and its original qualification remain in the
                # shared record. No failed expanded question becomes qualified.
                return base_terminal_design()
            raise ValueError('terminal questions unresolved; upstream recovery remains eligible: '
                             + record.get('error', 'author did not qualify the questions'))
        return parse_utility_design(record['design'])

    return resolve


def deferred_module_provider(scenario, spec, args, audio_config):
    """Create public module questions only if a later delivered attempt needs them."""
    registry = spec['deferred_module_designs']
    def resolve():
        import json
        record = shared_design(scenario, Path(registry['root']), args, audio_config,
                               registry['identity'], stage='modules')
        if record is None:
            return {}
        return {int(i): json.loads(Path(path).read_text())
                for i, path in record.get('module_designs', {}).items()}
    return resolve
