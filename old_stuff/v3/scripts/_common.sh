#!/usr/bin/env bash
# Shared helpers for user-facing workflows. Source this file; do not execute it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MBC_PYTHON:-$ROOT/.venv/bin/python}"

require_environment() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    cat >&2 <<EOF
Python environment not found at: $PYTHON_BIN
Create it with one of:
  ./scripts/bootstrap_cpu.sh
  ./scripts/bootstrap_cuda.sh
You may override the interpreter with MBC_PYTHON=/path/to/python.
EOF
    exit 2
  fi
}

run_module() {
  require_environment
  (
    cd "$ROOT"
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m "$@"
  )
}
