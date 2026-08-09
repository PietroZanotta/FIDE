#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --aggregate-existing|--no-plots) ARGS+=("$1"); shift;;
    --seeds) ARGS+=("$1" "$2"); shift 2;;
    -h|--help)
      "$PY" "$ROOT/stage3_rollout_adaptation.py" --help
      exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

banner "MFSI Stage 3: rollout-aware frozen-correction adaptation"
run_py "$ROOT/stage3_rollout_adaptation.py" "${ARGS[@]}"
