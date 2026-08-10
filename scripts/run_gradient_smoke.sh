#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
BACKEND="${MFSI_BACKEND:-tesseract}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    -h|--help) echo "Usage: ./scripts/run_gradient_smoke.sh [--backend tesseract|jax]"; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ "$BACKEND" == "jax" || "$BACKEND" == "tesseract" ]] || { echo "invalid backend" >&2; exit 2; }
if [[ "$BACKEND" == "jax" ]]; then
  # The served Tesseracts are CPU images. Force the direct reference onto CPU
  # as well so parity measures the component boundary, not GPU/CPU reduction
  # order in the long implicit-calibration solve.
  MFSI_BACKEND=jax JAX_PLATFORMS=cpu run_py \
    "$ROOT/gradient_backend_smoke.py" --backend jax
else
  JAX_PLATFORMS=cpu "$ROOT/scripts/_run_gradient_tesseracts.sh" \
    "$PY" "$ROOT/gradient_backend_smoke.py" --backend tesseract
fi
