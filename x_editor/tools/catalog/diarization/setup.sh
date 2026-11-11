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

# Prepare the pinned source with deployment/dependencies.py repos DiariZen first.
REPO_ROOT="$(cd ../../../../ && pwd)"
DIARIZEN_REPO="${DIARIZEN_REPO:-$REPO_ROOT/third_party/src/DiariZen}"
if [ ! -d "$DIARIZEN_REPO/pyannote-audio" ]; then
  echo "Set DIARIZEN_REPO to a prepared checkout, or run: python deployment/dependencies.py repos DiariZen (from the repository root)" >&2
  exit 1
fi
"$UV" sync --inexact
PY=".venv/bin/python"

"$UV" pip install --python "$PY" \
    -e "$DIARIZEN_REPO/pyannote-audio" \
    "torch==2.8.0" \
    "torchaudio==2.8.0" \
    "numpy==1.26.4" \
    "toml>=0.10" \
    "scipy>=1.10" \
    "soundfile>=0.12" \
    "huggingface_hub>=0.13" \
    "psutil>=5.9" \
    "accelerate==1.6.0"
"$UV" pip install --python "$PY" --no-deps -e "$DIARIZEN_REPO"
