"""Production review support extracted without changing function bodies."""

from __future__ import annotations


from copy import copy


import fcntl


import json


from pathlib import Path


from x_editor.revision.atomic_module_review import author_one, public_context, review_one


from x_editor.revision.refinement_goals import author_utility


from x_editor.evaluation.atomic_module_rubrics import bind_design


from x_editor.evaluation.iterative_runtime import sha256, write_record


def read(path):
    return json.loads(Path(path).read_text())


def shared_design(scenario, directory, args, audio_config, design_identity, *, stage='all'):
    """One candidate-blind author per task across all policy workers; never pick a winner."""
    from x_editor.revision.shared_job_lease import directory_lease
    try:
        with directory_lease(Path(directory)/scenario.fingerprint/'author.lease'):
            return _shared_design_owned(scenario, directory, args, audio_config, design_identity, stage=stage)
    except BlockingIOError:
        return None


def _shared_design_owned(scenario, directory, args, audio_config, design_identity, *, stage='all'):
    if stage not in {'all', 'modules', 'terminal'}:
        raise ValueError('invalid public authoring stage')
    directory = Path(directory)/scenario.fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'author.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        final = directory/'shared.json'
        if final.exists():
            value = read(final)
            if (value['design_identity'] != design_identity or value['scenario_fingerprint'] != scenario.fingerprint
                    or value.get('stage', 'all') != stage):
                raise ValueError('shared questions belong to another protocol')
            for path, digest in value.get('files', {}).items():
                if sha256(path) != digest:
                    raise ValueError('shared question design changed')
            return value
        # A crashed author leaves its complete raw attempt for audit. A new
        # attempt never overwrites those calls; only one completed decision wins.
        number = len(list(directory.glob('author_attempt_*'))) + 1
        root = directory/f'author_attempt_{number:04d}'
        root.mkdir()
        result = dict(protocol='x-editor.shared-public-utility/v1',
            scenario_fingerprint=scenario.fingerprint, design_identity=design_identity,
            candidate_visible_to_author=False, qualification_is_model_judgment=True,
            qualification_status='unqualified', module_designs={}, files={})
        if stage != 'all':
            result['stage'] = stage
        try:
            if number > 2:
                raise ValueError('interrupted source-author budget exhausted')
            author_args = copy(args)
            author_args.output = root/'modules'
            for index in (range(len(scenario.turns)) if stage != 'terminal' else []):
                context = public_context(scenario.model_dump(mode='json'), index)
                author_one(scenario.fingerprint, index, context, author_args)
                path = author_args.output/'cases'/scenario.fingerprint/'turns'/f'{index:04d}'/'design.json'
                if not path.exists():
                    revision_args = copy(author_args)
                    revision_args.draft_root = author_args.output
                    revision_args.output = root/'module_revision'
                    revision_args.wait_seconds = 0
                    review_one(scenario.fingerprint, index, context, revision_args)
                    path = revision_args.output/'cases'/scenario.fingerprint/'turns'/f'{index:04d}'/'design.json'
                bind_design(context, read(path))
                result['module_designs'][str(index)] = str(path)
                result['files'][str(path)] = sha256(path)
            if stage != 'modules':
                utility = author_utility(scenario, output=root/'terminal', args=args, audio_config=audio_config)
                result['design'] = utility.model_dump(mode='json')
            result.update(qualification_status='qualified', qualification_origin=str(root))
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error))
        write_record(final, result)
        return result
