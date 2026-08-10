#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
"$ROOT/scripts/run_gradient_smoke.sh" --backend jax
"$ROOT/scripts/run_gradient_smoke.sh" --backend tesseract
run_py "$ROOT/compare_gradient_backends.py"
