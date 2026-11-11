#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
UV="${X_EDITOR_UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV" ] || [ ! -x "$UV" ]; then
  echo "uv not found; install uv or set X_EDITOR_UV_BIN" >&2
  exit 1
fi
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export RUST_LOG="${X_EDITOR_UV_RUST_LOG:-warn}"
"$UV" venv .venv --python "${X_EDITOR_CLEARERVOICE_PYTHON:-python3.11}" --system-site-packages --allow-existing
"$UV" pip install --python .venv/bin/python -e .
if [ "${X_EDITOR_CLEARERVOICE_INSTALL_FULL_DEPS:-0}" = "1" ]; then
  "$UV" pip install --python .venv/bin/python "clearvoice==${X_EDITOR_CLEARVOICE_VERSION:-0.1.2}"
else
  echo "Installed ClearerVoice backend service wrapper only."
  echo "Set X_EDITOR_CLEARERVOICE_INSTALL_FULL_DEPS=1 to install clearvoice runtime deps into this .venv, or run service.py with an existing ClearVoice runtime Python."
fi
