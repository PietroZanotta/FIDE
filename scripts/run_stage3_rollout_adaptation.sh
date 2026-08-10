#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ARGS=()
BACKEND="${MFSI_BACKEND:-tesseract}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --legacy-output) ARGS+=("$1"); shift;;
    --aggregate-existing|--no-plots) ARGS+=("$1"); shift;;
    --seeds) ARGS+=("$1" "$2"); shift 2;;
    -h|--help)
      "$PY" "$ROOT/stage3_rollout_adaptation.py" --help
      exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

banner "MFSI Stage 3: rollout-aware frozen-correction adaptation"
"$ROOT/scripts/_run_backend_experiment.sh" stage3 "$BACKEND" "${ARGS[@]}"
