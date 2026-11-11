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

assert importlib.util.find_spec("requests") is not None
assert Path("client.py").exists()
assert Path("mcp_server.py").exists()
print("ace_step_music wrapper env ok")
PY

if [ "${ACE_STEP_REQUIRE_SERVICE:-0}" = "1" ]; then
  "$PY" - <<'PY'
import os
from client import AceStepMusicClient

url = os.getenv("ACE_STEP_API_URL") or os.getenv("ACE_STEP_UPSTREAM_URL") or os.getenv("ACE_STEP_SERVICE_URL")
client = AceStepMusicClient(url)
try:
    print(client.health())
    print(client.validate_api_compatibility())
except Exception as exc:
    raise SystemExit(f"ACE-Step service check failed for {client.base_url}: {type(exc).__name__}: {exc}")
PY
fi
