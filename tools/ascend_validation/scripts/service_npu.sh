#!/usr/bin/env bash
set -euo pipefail
CV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VLLM_VENDOR=ascend
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=0
export VLLM_FL_ASCEND_SLOT_MAPPING="${VLLM_FL_ASCEND_SLOT_MAPPING:-torch}"
cd "$CV_ROOT"
exec "$CV_ROOT/.venv/bin/python" "$CV_ROOT/service/control.py" "$@"
