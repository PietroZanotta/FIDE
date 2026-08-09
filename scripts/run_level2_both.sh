#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick|--no-plots) ARGS+=("$1"); shift;;
    -h|--help)
      echo "Usage: ./scripts/run_level2_both.sh [--quick] [--no-plots]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

"$ROOT/scripts/run_level2.sh" --backend jax "${ARGS[@]}"
"$ROOT/scripts/run_level2.sh" --backend tesseract "${ARGS[@]}"
run_py "$ROOT/scripts/compare_level2_backends.py"
