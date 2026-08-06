#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/run_comparison_seed_sweep.py \
  --config configs/scientific_seed_sweep.yaml \
  --output artifacts/registered_scientific_study
