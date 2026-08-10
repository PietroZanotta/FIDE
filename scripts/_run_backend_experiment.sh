#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
EXPERIMENT="$1"; BACKEND="$2"; shift 2
COMMAND=("$PY" "$ROOT/backend_experiment_runner.py" --experiment "$EXPERIMENT" --backend "$BACKEND" "$@")
if [[ "$BACKEND" == "jax" ]]; then
  MFSI_BACKEND=jax "${COMMAND[@]}"
elif [[ "$BACKEND" == "tesseract" ]]; then
  "$ROOT/scripts/_run_gradient_tesseracts.sh" "${COMMAND[@]}"
else
  echo "backend must be tesseract or jax" >&2; exit 2
fi
