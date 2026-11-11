#!/usr/bin/env bash
set -euo pipefail
: "${XEDIT_DATA:?Set XEDIT_DATA to a writable mounted deployment directory}"
TASK_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_SOURCE"
if ! command -v uv >/dev/null; then
  echo 'Prepare the base image with deployment/install_os.sh before launching services.' >&2
  exit 2
fi
if [[ "${1:-}" == "serve" ]]; then
  shift
  uv run --locked python deployment/bootstrap.py --data "$XEDIT_DATA" --phase start "$@"
  exec uv run --locked python deployment/supervise.py --data "$XEDIT_DATA"
fi
exec uv run --locked python deployment/bootstrap.py --data "$XEDIT_DATA" "$@"
