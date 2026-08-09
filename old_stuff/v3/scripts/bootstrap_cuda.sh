#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECREATE=0
if [[ "${1:-}" == "--recreate" ]]; then
  RECREATE=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--recreate]" >&2
  exit 2
fi

PYTHON_BOOTSTRAP="${MBC_BOOTSTRAP_PYTHON:-python3}"
if [[ $RECREATE -eq 1 ]]; then
  rm -rf .venv
fi
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BOOTSTRAP" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements/host-cuda.txt
.venv/bin/python -m pip install -e . --no-build-isolation

echo "CUDA environment ready: $ROOT/.venv"
echo "Run ./scripts/run_tests.sh or ./scripts/run_from_scratch_acceptance.sh"
