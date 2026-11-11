#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python - <<'PY'
import importlib.util

for name in ["fastapi", "uvicorn", "audio_separator", "onnxruntime"]:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing dependency: {name}")
print("audio_separator_roformer env ok")
PY
