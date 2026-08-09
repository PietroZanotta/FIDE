#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEED=20260806
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="artifacts/diffpop_full_seed_${SEED}"
fi
./scripts/run_comparison.sh \
  --config configs/diffpop_full.yaml \
  --output "$OUTPUT" \
  --seed "$SEED"
