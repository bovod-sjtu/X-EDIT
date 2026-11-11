#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${X_EDITOR_INSTALL_GPURIR:-0}" == "1" ]]; then
  if ! command -v nvcc >/dev/null 2>&1 && [[ -z "${CUDACXX:-}" ]]; then
    for cuda_dir in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.2 /usr/local/cuda-12.6; do
      if [[ -n "$cuda_dir" && -x "$cuda_dir/bin/nvcc" ]]; then
        export CUDA_HOME="$cuda_dir"
        export CUDA_PATH="$cuda_dir"
        export CUDACXX="$cuda_dir/bin/nvcc"
        export PATH="$cuda_dir/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_dir/lib64:${LD_LIBRARY_PATH:-}"
        break
      fi
    done
  fi
  uv sync --extra gpurir
else
  uv sync
fi
