#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python"
fi

"$PY" - <<'PY'
import importlib.util
import shutil
from pathlib import Path

required = ["fastapi", "uvicorn", "requests", "pydantic"]
for name in required:
    assert importlib.util.find_spec(name) is not None, f"missing required package: {name}"

for filename in ["service.py", "client.py", "mcp_server.py", "capability.yaml", "manual.yaml", "config.yaml"]:
    assert Path(filename).exists(), f"missing {filename}"

print("seamless_expressive_s2st wrapper env ok")
print(f"seamless_communication importable: {importlib.util.find_spec('seamless_communication') is not None}")
print(f"m4t_predict cli: {shutil.which('m4t_predict') or 'missing'}")
print(f"expressivity_predict cli: {shutil.which('expressivity_predict') or 'missing'}")
PY

if [ "${REQUIRE_SEAMLESS_RUNTIME:-0}" = "1" ]; then
  command -v m4t_predict >/dev/null
  command -v expressivity_predict >/dev/null
  test -n "${SEAMLESS_EXPRESSIVE_MODEL_DIR:-}"
  test -d "$SEAMLESS_EXPRESSIVE_MODEL_DIR"
fi
