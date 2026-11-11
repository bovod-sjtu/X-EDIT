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

"$UV" run --project . python - <<'PY'
import importlib.util
from pathlib import Path

assert importlib.util.find_spec("fastapi") is not None
assert importlib.util.find_spec("uvicorn") is not None
assert Path("service.py").exists()
print("diarization service wrapper env ok")
PY
