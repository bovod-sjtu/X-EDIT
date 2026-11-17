"""Launch a verified public pipeline entrypoint without changing upstream code."""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from pipeline import ROOT, SPEC, inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['text', 'prosody', 'emotion', 'pause'])
    args, service_args = parser.parse_known_args()
    inspect()
    sys.path.insert(0, str(ROOT / json.loads(SPEC.read_text())['path']))
    module = f'speech_edit.pipelines.{args.task}.service.api_service'
    sys.argv = [module, *service_args]
    importlib.import_module(module).main()


if __name__ == '__main__':
    main()
