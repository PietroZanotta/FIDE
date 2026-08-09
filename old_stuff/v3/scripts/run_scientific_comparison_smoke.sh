#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="artifacts/diffpop_smoke"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
./scripts/run_comparison.sh \
  --config configs/diffpop_smoke.yaml \
  --output "$OUTPUT" \
  "${ARGS[@]}"
