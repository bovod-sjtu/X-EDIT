#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python"
fi
"$PY" - <<'PY'
import importlib.util
from pathlib import Path
assert importlib.util.find_spec("pydub") is not None
assert Path("mcp_server.py").exists()
print("paralinguistic_event_bank env ok")
PY
