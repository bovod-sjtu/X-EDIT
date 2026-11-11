#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
UV="${X_EDITOR_UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV" ] || [ ! -x "$UV" ]; then
  echo "uv not found; install uv or set X_EDITOR_UV_BIN" >&2
  exit 1
fi
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
export RUST_LOG="${X_EDITOR_UV_RUST_LOG:-warn}"
"$UV" venv .venv --python "${X_EDITOR_QUARKAUDIO_PYTHON:-python3.11}" --system-site-packages --allow-existing
"$UV" pip install --python .venv/bin/python -e .
if [ "${X_EDITOR_QUARKAUDIO_INSTALL_FULL_DEPS:-0}" = "1" ]; then
  missing="$(
    .venv/bin/python - <<'PY'
import importlib.util

packages = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "pytorch_lightning": "pytorch-lightning",
    "transformers": "transformers",
    "soundfile": "soundfile",
    "librosa": "librosa",
    "numpy": "numpy",
    "scipy": "scipy",
    "omegaconf": "omegaconf",
    "safetensors": "safetensors",
    "einops": "einops",
    "einx": "einx",
    "soxr": "soxr",
    "tqdm": "tqdm",
    "x_transformers": "x-transformers",
}
print(" ".join(package for module, package in packages.items() if importlib.util.find_spec(module) is None))
PY
  )"
  if [ -n "$missing" ]; then
    "$UV" pip install --python .venv/bin/python --no-deps $missing
  else
    echo "QuarkAudio-UniSE runtime imports already available in this backend venv."
  fi
else
  echo "Installed QuarkAudio-UniSE backend service wrapper only."
  echo "Set X_EDITOR_QUARKAUDIO_INSTALL_FULL_DEPS=1 to install missing model runtime dependencies."
fi
