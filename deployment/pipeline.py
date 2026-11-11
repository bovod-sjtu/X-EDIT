"""Fetch and verify the pipeline and its official SDK submodules at release pins.

Works in Git checkouts and source archives. Never applies patches to an official
SDK: the upstream service entrypoints own their isolated execution adapters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'third_party/external_pipeline.json'


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def inspect(tools=(), *, fetch=False):
    spec = json.loads(SPEC.read_text())
    if set(tools) & set(spec.get('external_tools', [])):
        raise ValueError('请自行下载并安装F5-TTS; the SDK and weights are operator-installed only, outside this checkout')
    path = ROOT / spec['path']
    rows = {Path(r['path']).parts[1]: r for r in spec['submodules']}
    if set(tools) - rows.keys():
        raise ValueError('Unknown pipeline tools: ' + str(sorted(set(tools) - rows.keys())))
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    if not (path / '.git').exists():
        if not fetch:
            raise RuntimeError('Pipeline missing; run deployment/pipeline.py fetch')
        if path.exists() and any(path.iterdir()):
            raise RuntimeError('Refusing to replace an existing non-Git pipeline directory')
        subprocess.run(['git', 'clone', '--no-checkout', spec['url'], str(path)], check=True, env=env)
        subprocess.run(['git', '-C', str(path), 'checkout', '--detach', spec['commit']], check=True, env=env)
    if git(path, 'rev-parse', 'HEAD') != spec['commit']:
        raise RuntimeError('Pipeline pin differs; inspect changes before an explicit update')
    if git(path, 'status', '--porcelain', '--ignore-submodules=all'):
        raise RuntimeError('Pipeline outer source has local changes')
    checked = []
    for name in sorted(set(tools)):
        row = rows[name]
        source = path / row['path']
        tracked = git(path, 'ls-tree', 'HEAD', row['path']).split()
        if len(tracked) < 3 or tracked[0] != '160000' or tracked[2] != row['commit']:
            raise RuntimeError('SDK gitlink differs from release manifest: ' + name)
        if fetch:
            if (source / '.git').exists() and (
                git(source, 'rev-parse', 'HEAD') != row['commit']
                or git(source, 'status', '--porcelain')
            ):
                raise RuntimeError('Refusing to update a changed SDK: ' + name)
            # Also finish an interrupted recursive checkout (e.g. F5/BigVGAN).
            subprocess.run(['git', '-C', str(path), 'submodule', 'update', '--init', '--recursive', '--', row['path']], check=True, env=env)
        if not (source / '.git').exists():
            raise RuntimeError('SDK missing: ' + name + '; run pipeline.py fetch ' + name)
        if git(source, 'rev-parse', 'HEAD') != row['commit']:
            raise RuntimeError('Wrong SDK pin: ' + name)
        if git(source, 'status', '--porcelain'):
            raise RuntimeError('Official SDK has local changes: ' + name)
        nested = git(source, 'submodule', 'status', '--recursive')
        if any(line.startswith(('-', '+', 'U')) for line in nested.splitlines()):
            raise RuntimeError('Nested SDK source is missing or at a different pin: ' + name)
        adapter_path = source.parent / 'upstream.json'
        if adapter_path.is_file():
            adapter = json.loads(adapter_path.read_text())
            if adapter['commit'] != row['commit']:
                raise RuntimeError('Adapter pin differs: ' + name)
            for patch in adapter.get('patches', []):
                patch_path = source.parent / patch.get('path', patch.get('file', ''))
                if hashlib.sha256(patch_path.read_bytes()).hexdigest() != patch['sha256']:
                    raise RuntimeError('Upstream patch checksum differs: ' + str(patch_path))
        checked.append(name)
    return {'commit': spec['commit'], 'checked_tools': checked,
            'official_sources_clean': True, 'inference_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fetch', 'check'])
    parser.add_argument('tools', nargs='*', help='Empty: pipeline only; all: managed SDKs (excludes operator-installed tools)')
    args = parser.parse_args()
    tools = json.loads(SPEC.read_text())['required_tools'] if args.tools == ['all'] else args.tools
    print(json.dumps(inspect(tools, fetch=args.action == 'fetch'), indent=2))


if __name__ == '__main__':
    main()
