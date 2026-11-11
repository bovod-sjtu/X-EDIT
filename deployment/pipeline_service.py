"""Launch a pinned pipeline API with Ubuntu 22.04 Git compatibility.

Only the artifact-boundary query is adapted. No editing/model code or official
checkout is changed. Git versions supporting NUL worktree output use upstream.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import importlib
import subprocess
import sys

from pipeline import ROOT, SPEC, inspect


def activate_paths_compat():
    from speech_edit.runtime import paths
    original = paths._git_context

    def context(project):
        env = {k: v for k, v in os.environ.items()
               if k not in {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE'}}
        def git(*args):
            return subprocess.run(['git', '-C', str(project), *args], env=env,
                                  capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        probe = subprocess.run(['git', '-C', str(project), 'worktree', 'list', '--porcelain', '-z'],
                               env=env, capture_output=True, timeout=5)
        if probe.returncode != 129:  # Not an unsupported-option error.
            return original(project)
        if Path(git('rev-parse', '--show-toplevel')).resolve() != project:
            return original(project)
        git_dir = Path(git('rev-parse', '--absolute-git-dir')).resolve()
        common = Path(git('rev-parse', '--git-common-dir'))
        common = (project / common).resolve() if not common.is_absolute() else common.resolve()
        roots = {project, common}
        for field in git('worktree', 'list', '--porcelain').splitlines():
            if field.startswith('worktree '):
                value = field[len('worktree '):]
                if value.startswith('"'):
                    value = json.loads(value)
                if '\n' in value or '\r' in value or not Path(value).is_absolute():
                    raise ValueError('Unsupported worktree path for legacy Git; upgrade Git')
                roots.add(Path(value).resolve())
        return git_dir.relative_to(common).as_posix(), tuple(roots)

    paths._git_context = context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['text', 'prosody', 'emotion', 'pause'])
    args, service_args = parser.parse_known_args()
    inspect()
    sys.path.insert(0, str(ROOT / json.loads(SPEC.read_text())['path']))
    activate_paths_compat()
    module = f'speech_edit.pipelines.{args.task}.service.api_service'
    sys.argv = [module, *service_args]
    # Keep request-model annotations bound to their real module for Pydantic.
    importlib.import_module(module).main()


if __name__ == '__main__':
    main()
