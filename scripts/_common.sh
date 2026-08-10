#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -d "$ROOT/.venv/bin" ]]; then
  export PATH="$ROOT/.venv/bin:$PATH"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
if [[ -n "${MFSI_PYTHON:-}" ]]; then PY="$MFSI_PYTHON";
elif [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python";
else PY="$(command -v python3 || command -v python)"; fi
[[ -x "$PY" ]] || { echo "Python not found; run ./scripts/install.sh" >&2; exit 2; }
mkdir -p "$ROOT/results/logs"
run_py(){ "$PY" "$@"; }
banner(){ printf '\n============================================================\n%s\n============================================================\n' "$*"; }
