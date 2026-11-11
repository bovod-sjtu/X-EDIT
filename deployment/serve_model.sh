#!/usr/bin/env bash
set -euo pipefail
# Run in the isolated model-server environment. Explicit paths prevent ref drift.
case "${1:?usage: serve_model.sh planner|omni}" in
planner)
  : "${XEDIT_PLANNER_MODEL_PATH:?set a pinned snapshot path}"
  exec vllm serve "$XEDIT_PLANNER_MODEL_PATH" --served-model-name Qwen/Qwen3.8-27B \
    --host 127.0.0.1 --port 8040 --tensor-parallel-size "${XEDIT_PLANNER_TP:-4}" \
    --max-model-len 65536 --max-num-seqs 2 --gpu-memory-utilization "${XEDIT_PLANNER_GPU_FRACTION:-0.50}" \
    --enable-prefix-caching --trust-remote-code --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --enforce-eager \
    --disable-custom-all-reduce --gdn-prefill-backend triton
  ;;
omni)
  : "${XEDIT_OMNI_MODEL_PATH:?set a pinned snapshot path}"
  exec vllm serve "$XEDIT_OMNI_MODEL_PATH" --served-model-name Qwen3Omni-Instruct \
    --host 127.0.0.1 --port 8011 --trust-remote-code --max-model-len 16384 \
    --max-num-seqs 1 --gpu-memory-utilization "${XEDIT_OMNI_GPU_FRACTION:-0.90}"
  ;;
*) echo "Choose planner or omni" >&2; exit 2 ;;
esac
