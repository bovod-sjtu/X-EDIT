#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

cat <<'MSG'
ACE-Step X-Editor wrapper installed.

This installs only the lightweight wrapper. To run the real model, install and
start the official ACE-Step 1.5 API separately:

  git clone https://github.com/ace-step/ACE-Step-1.5.git /path/to/ACE-Step-1.5
  cd /path/to/ACE-Step-1.5
  uv sync
  uv run acestep-api

Then set:

  export ACE_STEP_API_URL=http://127.0.0.1:8044
MSG
