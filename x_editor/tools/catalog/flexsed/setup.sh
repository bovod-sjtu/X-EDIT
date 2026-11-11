#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${XEDITOR_PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "No Python interpreter found for FlexSED environment setup." >&2
  exit 1
fi

UV_BIN="${X_EDITOR_UV_BIN:-$(command -v uv || true)}"
[[ -x "$UV_BIN" ]] || { echo "Install uv or set X_EDITOR_UV_BIN." >&2; exit 1; }
# uv does not require the image's optional ensurepip package. Keep the complete
# dependency set isolated: a venv created from another venv does not inherit it.
"$UV_BIN" venv --allow-existing --python "$PYTHON_BIN" .venv
"$UV_BIN" pip sync --python .venv/bin/python requirements.lock
"$UV_BIN" pip check --python .venv/bin/python

if [[ -z "${FLEXSED_REPO_DIR:-}" ]]; then
  echo "Set FLEXSED_REPO_DIR to an official JHU-LCAP/FlexSED checkout before real inference." >&2
fi
