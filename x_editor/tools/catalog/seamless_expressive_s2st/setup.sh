#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

cat <<'EOF'
Wrapper environment installed.

To enable real model inference, install Meta seamless_communication in this
same environment and provide approved SeamlessExpressive artifacts:

  git clone https://github.com/facebookresearch/seamless_communication.git /path/to/seamless_communication
  .venv/bin/python -m pip install -e /path/to/seamless_communication
  export SEAMLESS_EXPRESSIVE_MODEL_DIR=/path/to/approved/SeamlessExpressive/model

SeamlessExpressive artifacts are gated and are not downloaded by this setup
script. See README.md for license and access notes.
EOF
