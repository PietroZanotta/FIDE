#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
rm -rf artifacts/diffpop_micro
PYTHONPATH=src python -m manybody_completion.comparison_cli \
  --config configs/diffpop_micro.yaml \
  --output artifacts/diffpop_micro
PYTHONPATH=src python scripts/run_composed_gradient_probe.py \
  --config configs/diffpop_micro.yaml \
  --output artifacts/diffpop_micro/composed_gradient_probe.json
PYTHONPATH=src python scripts/validate_scientific_outputs.py \
  --directory artifacts/diffpop_micro
