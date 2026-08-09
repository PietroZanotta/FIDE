#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src python scripts/run_comparison_seed_sweep.py \
  --config configs/scientific_seed_sweep.yaml
