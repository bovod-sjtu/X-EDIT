#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
from x_editor.tools.catalog.edit_data_pipeline.client import DEFAULT_URLS, EditDataPipelineClient
from x_editor.tools.edit_backend import selected_edit_backend

client = EditDataPipelineClient()
print("Tagged-edit backend:", selected_edit_backend())
passed = True
for name in DEFAULT_URLS:
    try:
        health = client.health(name)
        ok = health.get("status") in {"ok", "healthy", "ready", "running", "loaded"}
        passed = passed and ok
        print(f"{name}: {'OK' if ok else 'BAD'} health={health}")
    except Exception as exc:
        passed = False
        print(f"{name}: BAD {type(exc).__name__}: {exc}")
raise SystemExit(0 if passed else 1)
PY
