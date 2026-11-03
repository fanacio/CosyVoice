#!/usr/bin/env bash
set -euo pipefail
CV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_VENDOR=ascend
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=0
export VLLM_FL_ASCEND_SLOT_MAPPING="${VLLM_FL_ASCEND_SLOT_MAPPING:-torch}"
cd "$CV_ROOT"
exec "$CV_ROOT/.venv/bin/python" -u "$CV_ROOT/scripts/run_tts_npu.py" --root "$CV_ROOT" "$@"
