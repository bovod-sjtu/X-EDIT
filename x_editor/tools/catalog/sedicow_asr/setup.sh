#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create the SE-DiCoW service environment" >&2
  exit 1
fi

uv sync --inexact

if [ "${SEDICOW_CLONE_OFFICIAL:-0}" = "1" ]; then
  repo_dir="${SEDICOW_REPO:-$(pwd)/vendor/DiCoW}"
  mkdir -p "$(dirname "$repo_dir")"
  if [ ! -d "$repo_dir/.git" ]; then
    git clone https://github.com/BUTSpeechFIT/DiCoW "$repo_dir"
  fi
  echo "Official DiCoW checkout: $repo_dir"
fi

echo "SE-DiCoW service wrapper env ok"
