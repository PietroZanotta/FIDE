#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --aggregate-existing|--no-plots) ARGS+=("$1"); shift;;
    -h|--help) "$PY" "$ROOT/stage3b_confirmatory.py" --help; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

banner "MFSI Stage 3B: confirmatory rollout credit assignment"
run_py "$ROOT/stage3b_confirmatory.py" "${ARGS[@]}"
