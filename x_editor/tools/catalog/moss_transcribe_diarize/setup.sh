#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
uv venv --python "${PYTHON:-python3}" .venv
.venv/bin/python -m pip install -U pip
uv pip install --python .venv/bin/python -e .
