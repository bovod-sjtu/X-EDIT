#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
.venv/bin/python - <<'PY'
import fastapi
import pydantic
import uvicorn
print("sedicow_asr service wrapper env ok")
PY
