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
    --skip-backend-parity) shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$OUTPUT" ]]; then OUTPUT="artifacts/diffpop_full_seed_${SEED}"; fi
PYTHONPATH=src python -m manybody_completion.comparison_cli \
  --config configs/diffpop_full.yaml --seed "$SEED" --output "$OUTPUT"
PYTHONPATH=src python scripts/validate_scientific_outputs.py --directory "$OUTPUT"
