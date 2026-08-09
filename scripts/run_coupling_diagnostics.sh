#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-plots|--reuse-computed) ARGS+=("$1"); shift;;
    --resamples|--pairs-per-time|--mmd-resamples|--output-dir)
      ARGS+=("$1" "$2"); shift 2;;
    -h|--help)
      "$PY" "$ROOT/diagnose_coupling_stage2.py" --help; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

banner "MFSI Stage 2 coupling diagnosis (no optimization)"
run_py "$ROOT/diagnose_coupling_stage2.py" "${ARGS[@]}"
run_py "$ROOT/validate_coupling_diagnostics.py"
