"""Report release blockers across the declared service dependency graph.

This is a fail-closed review aid, not a legal compatibility certificate. Source,
model, asset and external pipeline declarations remain separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def classify(declaration, permitted):
    normalized = str(declaration or '').lower()
    if ';' in normalized:
        return 'mixed_license_scope_needs_review'
    if 'by-nc' in normalized or 'research' in normalized:
        return 'noncommercial_or_research_restriction'
    if normalized in {value.lower() for value in permitted}:
        return 'permissive_declaration'
    if normalized in {'gpl-3.0', 'gpl-3.0-only', 'gpl-3.0-or-later'}:
        return 'copyleft_obligations_need_review'
    if normalized == 'cc-by-4.0':
        return 'attribution_obligations_need_review'
    return 'license_scope_needs_review'


def audit(root=ROOT):
    def read(name):
        return json.loads((root / name).read_text())

    manifest = read('third_party/manifest.json')
    inventory = read('third_party/licenses.json')
    pipeline = read('third_party/external_pipeline.json')
    policy = read('third_party/release_review.json')
    stack = read('deployment/stack.json')
    entries = {}
    for row in inventory['entries']:
        key = (row['kind'], row['name'])
        if key in entries:
            raise ValueError('Duplicate license inventory identity: ' + str(key))
        entries[key] = row
    overrides = {(row['kind'], row['name']): row['reason'] for row in policy['review_overrides']}
    models = {row['repo_id']: row for row in manifest['models']}
    sources = {row['name']: row for row in manifest['repositories']}
    sdk_rows = {Path(row['path']).parts[1]: row for row in pipeline['submodules']}
    pipeline_license = pipeline.get('license', {})
    license_file = root / pipeline['path'] / pipeline_license.get('file', 'LICENSE')
    pipeline_license_verified = (pipeline_license.get('revision') == pipeline['commit']
                                 and license_file.is_file()
                                 and hashlib.sha256(license_file.read_bytes()).hexdigest() == pipeline_license.get('sha256'))
    dependencies = {}

    def dependency(kind, name, revision=None):
        identity = kind + ':' + name
        if identity not in dependencies:
            row = entries.get((kind, name), {})
            declaration = row.get('license_summary')
            status = classify(declaration, policy['permissive_declarations'])
            reason = overrides.get((kind, name))
            pin_matches = revision is None or row.get('revision') == revision
            if not row or not pin_matches:
                status = 'missing_or_mismatched_license_evidence'
            elif reason:
                status = 'component_review_required'
            dependencies[identity] = {
                'kind': kind, 'name': name, 'revision': revision,
                'declaration': declaration, 'status': status,
                'reason': reason, 'pin_matches': pin_matches,
                'evidence': row.get('documents', []),
            }
        return identity

    direct = {}
    for name, service in stack['services'].items():
        required = []
        for model in service['models']:
            required.append(dependency('model', model, models[model]['revision']))
        for source in service.get('repositories', []):
            required.append(dependency('source', source, sources[source]['commit']))
        for group in service.get('service_assets', []):
            for row in manifest.get('service_assets', []):
                if Path(row['file']).parts[1] == group:
                    required.append(dependency('service_asset', row['file']))
        if service.get('pipeline_required'):
            identity = 'pipeline:root'
            dependencies[identity] = {
                'kind': 'pipeline', 'name': 'edit_data_pipeline',
                'revision': pipeline['commit'],
                'status': classify(pipeline_license.get('spdx'), policy['permissive_declarations']) if pipeline_license_verified else 'missing_or_mismatched_license_evidence',
                'declaration': pipeline_license.get('spdx'), 'reason': pipeline['license_note'],
            }
            required.append(identity)
        for tool in service.get('pipeline_tools', []) + service.get('external_pipeline_tools', []):
            identity = 'pipeline_sdk:' + tool
            evidence = policy['pipeline_sdk_declarations'].get(tool, {})
            declaration = evidence.get('license')
            pin = sdk_rows[tool]['commit']
            status = classify(declaration, policy['permissive_declarations'])
            if evidence.get('revision') != pin:
                status = 'missing_or_mismatched_license_evidence'
            dependencies[identity] = {
                'kind': 'pipeline_sdk', 'name': tool,
                'revision': pin,
                'declaration': declaration,
                'status': status,
                'evidence_url': evidence.get('evidence_url'),
            }
            required.append(identity)
        direct[name] = set(required)

    resolved = {}

    def collect(name, visiting):
        if name in visiting:
            raise ValueError('Cyclic service dependency: ' + name)
        if name not in resolved:
            required = set(direct[name])
            for parent in stack['services'][name]['depends_on']:
                required.update(collect(parent, visiting | {name}))
            resolved[name] = required
        return resolved[name]

    services = []
    for name in stack['services']:
        required = collect(name, set())
        blockers = sorted(identity for identity in required
                          if dependencies[identity]['status'] != 'permissive_declaration')
        services.append({'service': name, 'dependencies': sorted(required), 'blockers': blockers,
                         'operator_installed_only': bool(stack['services'][name].get('external_only')),
                         'declared_dependencies_clear': not blockers})
    for identity, dependency_row in dependencies.items():
        dependency_row['affected_services'] = sorted(name for name, required in resolved.items()
                                                    if identity in required)
    blockers = [row for row in dependencies.values() if row['status'] != 'permissive_declaration']
    return {
        'schema_version': 'x-edit.license-audit/v1',
        'review_date': policy['review_date'],
        'full_stack_approved': policy['full_stack_approved'],
        'full_stack_review_clear': not blockers and not policy['unreviewed_scope'],
        'source_release_assessed': False,
        'blockers': blockers, 'services': services,
        'replacement_candidates': policy['replacement_candidates'],
        'unreviewed_scope': policy['unreviewed_scope'],
        'inference_verified': False,
        'scope': 'Declared source/model/asset and service graph. External HTTP and separate downloads do not clear usage restrictions. GPL is not a noncommercial license.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'full_stack_review_clear': result['full_stack_review_clear'],
                      'source_release_assessed': result['source_release_assessed'],
                      'blockers': [{key: row.get(key) for key in
                                    ('kind', 'name', 'status', 'affected_services')}
                                   for row in result['blockers']],
                      'unreviewed_scope': result['unreviewed_scope']}, indent=2))
    raise SystemExit(0 if result['full_stack_review_clear'] else 2)


if __name__ == '__main__':
    main()
