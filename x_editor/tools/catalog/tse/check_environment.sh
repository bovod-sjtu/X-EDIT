#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="../quarkaudio_unise/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python"
fi
"$PY" - <<'PY'
from pathlib import Path
assert Path("mcp_server.py").exists()
assert Path("../quarkaudio_unise/client.py").exists()
print("tse mcp wrapper ok")
PY
