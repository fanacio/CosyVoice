#!/usr/bin/env bash
set -euo pipefail
CV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CV_ROOT"
exec "$CV_ROOT/.venv/bin/python" "$CV_ROOT/service/client.py" "$@"
