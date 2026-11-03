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

if [[ ! -x "$CV_ROOT/.venv/bin/python" ]]; then
    printf 'Missing project Python environment: %s\n' "$CV_ROOT/.venv" >&2
    exit 1
fi
exec "$CV_ROOT/.venv/bin/python" -u "$CV_ROOT/scripts/run_tts.py" --root "$CV_ROOT" "$@"
