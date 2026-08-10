#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
BACKEND="${MFSI_BACKEND:-tesseract}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --legacy-output) ARGS+=("$1"); shift;;
    --output-dir) ARGS+=("$1" "$2"); shift 2;;
    -h|--help)
      echo "Usage: ./scripts/run_stage4b_confirmatory.sh [--backend tesseract|jax] [--legacy-output] [--output-dir DIR]"
      exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
banner "MFSI Stage 4B: confirmatory fiber design (gradient backend: $BACKEND)"
"$ROOT/scripts/_run_backend_experiment.sh" stage4b "$BACKEND" "${ARGS[@]}"
