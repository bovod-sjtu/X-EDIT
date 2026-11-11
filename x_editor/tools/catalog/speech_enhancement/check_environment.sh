#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="../clearer_voice_studio/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python"
fi
"$PY" - <<'PY'
from pathlib import Path
assert Path("mcp_server.py").exists()
assert Path("../clearer_voice_studio/client.py").exists()
print("speech_enhancement mcp wrapper ok")
PY
