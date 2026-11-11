"""Audit a source release without downloading models or contacting services.

This verifies source/package wiring. It is not an inference or legal certificate.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', '__pycache__', '.cache', 'build', 'dist',
            'runs', 'models', 'checkpoints', '.pytest_cache', '.ruff_cache'}


def source_files():
    for directory, dirs, files in os.walk(ROOT):
        base = Path(directory)
        dirs[:] = [d for d in dirs if d not in EXCLUDED
                   and not d.endswith('.egg-info')
                   and base / d not in {ROOT / 'third_party/src', ROOT / 'third_party/dots_tts_edit'}]
        for name in files:
            if name == '.env' or (name.startswith('.env.') and name != '.env.example'):
                continue
            if name.endswith(('.pyc', '.pyo')):
                continue
            yield base / name


def audit():
    failures = []
    checks = []

    def check(name, ok, detail=None):
        row = {'check': name, 'passed': bool(ok)}
        if detail is not None:
            row['detail'] = detail
        checks.append(row)
        if not ok:
            failures.append(name)

    read = lambda name: json.loads((ROOT / name).read_text())
    manifest = read('third_party/manifest.json')
    stack = read('deployment/stack.json')
    policies = read('x_editor/tools/external_services.json')
    contracts = read('deployment/service_contracts.json')
    for name, policy in policies.items():
        service = stack['services'][name]
        check('operator-installed only: ' + name,
              service.get('external_only') and not service['environment']
              and not service['entry'] and not service['args']
              and not service.get('pipeline_tools') and not service.get('pipeline_required'))
        check('disabled by default: ' + name,
              policy['enable_env'] + '=0' in (ROOT / '.env.example').read_text())
    for name in ['sam_audio', 'voice_sculptor']:
        check('documentation-only extension: ' + name,
              name not in stack['services'] and name not in policies
              and all(spec['name'] != name for spec in contracts)
              and not any((ROOT / 'x_editor/tools/catalog' / name).glob('*.py'))
              and not any((ROOT / 'x_editor/tools/catalog' / name).glob('*.yaml')))
    sources = {r['name']: r for r in manifest['repositories']}
    models = {r['repo_id']: r for r in manifest['models']}
    for name, row in sources.items():
        check('source pin: ' + name, bool(re.fullmatch('[0-9a-f]{40}', row['commit'])))
    for name, row in models.items():
        check('model pin: ' + name, bool(re.fullmatch('[0-9a-f]{40}', row['revision'])))
    for name, env in stack['environments'].items():
        check('environment lock: ' + name, (ROOT / env['lock']).is_file())
    for name, service in stack['services'].items():
        check('service references: ' + name,
              (service['environment'] in stack['environments'] or
               (service.get('external_only') and service['environment'] is None and not service['entry'] and not service['args']))
              and all(d in stack['services'] for d in service['depends_on'])
              and all(m in models for m in service['models'])
              and all(r in sources for r in service.get('repositories', [])))
        for entry in service['entry']:
            if entry.startswith('{root}/'):
                check('service entry: ' + name, (ROOT / entry[7:]).is_file())
    check('unique ports', len({s['port'] for s in stack['services'].values()})
          == len(stack['services']))
    check('no duplicate backend adapters', not (ROOT / 'backends').exists())
    try:
        from pipeline import inspect
        result = inspect()
        check('pinned clean pipeline', True, result)
    except (RuntimeError, OSError, ValueError) as error:
        check('pinned clean pipeline', False, str(error))
    export = read('SOURCE_EXPORT.json')
    for row in export['shared_files'] + export['new_files']:
        path = ROOT / row['file']
        check('export hash: ' + row['file'], path.is_file()
              and hashlib.sha256(path.read_bytes()).hexdigest() == row['release_sha256'])
    parse_errors = []
    core_backend_imports = []
    nonportable_resources = []
    private_resource = re.compile(
        r'/(?:prodcpfs|cpfs|oss)/|\.codex/(?:shared|worktrees)/'
        r'|https?://[^\s\"\x27]*(?:devops\.|\.internal(?:/|:))'
        r'|https?://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+'
        r'|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?=[:/\s])'
    )
    python_count = 0
    for path in source_files():
        if path.is_relative_to(ROOT / 'x_editor') and path.suffix in {'.py', '.yaml', '.json', '.sh'}:
            if private_resource.search(path.read_text()):
                nonportable_resources.append(str(path.relative_to(ROOT)))
        if path.suffix != '.py':
            continue
        python_count += 1
        try:
            tree = ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        except (SyntaxError, UnicodeError) as error:
            parse_errors.append(str(path.relative_to(ROOT)) + ': ' + str(error))
            continue
        if path.is_relative_to(ROOT / 'x_editor') and 'catalog' not in path.parts:
            for node in ast.walk(tree):
                imports = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                           else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                if any(n.split('.')[0] in {'indextts', 'seed_vc', 'backends'} for n in imports):
                    core_backend_imports.append(str(path.relative_to(ROOT)))
    check('Python syntax', not parse_errors, {'files': python_count, 'errors': parse_errors})
    check('controller does not import restricted backends', not core_backend_imports,
          sorted(set(core_backend_imports)))
    check('no cluster paths or private default endpoints in runtime source',
          not nonportable_resources, sorted(set(nonportable_resources)))
    data_dir = ROOT / 'x_editor/accent_library/data'
    reference_files = [str(p.relative_to(ROOT)) for p in data_dir.rglob('*')
                       if p.is_file() and p != data_dir / 'README.md']
    check('no bundled accent reference data', not reference_files, reference_files)
    return {'schema_version': 'x-edit.source-audit/v1', 'passed': not failures,
            'service_count': len(stack['services']), 'environment_count': len(stack['environments']),
            'inference_verified': False, 'failures': failures, 'checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
