#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi
"$PY" - <<'PY'
from pathlib import Path
import os
source = Path(os.environ.get("QUARKAUDIO_UNISE_REPO_DIR", "vendor/QuarkAudio-UniSE")).expanduser()
for filename in [Path("service.py"), Path("client.py"), source / "test.py"]:
    assert filename.exists(), filename
print("quarkaudio_unise backend wrapper ok")
PY
