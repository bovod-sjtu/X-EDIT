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
for name in ["fastapi", "uvicorn", "requests", "numpy", "soundfile"]:
    assert importlib.util.find_spec(name) is not None, name
if __import__("os").getenv("X_EDITOR_CLEARERVOICE_REQUIRE_RUNTIME", "0") not in {"0", "false", "False"}:
    assert importlib.util.find_spec("clearvoice") is not None, "clearvoice"
assert Path("service.py").exists()
assert Path("client.py").exists()
print("clearer_voice_studio backend env ok")
PY
