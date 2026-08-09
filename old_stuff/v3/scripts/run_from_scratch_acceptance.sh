#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="artifacts/diffpop_micro"
rm -rf "$OUTPUT"
./scripts/run_comparison.sh \
  --config configs/diffpop_micro.yaml \
  --output "$OUTPUT"
./scripts/run_composed_gradient_probe.sh \
  --config configs/diffpop_micro.yaml \
  --output "$OUTPUT/composed_gradient_probe.json" \
  --experiment-report "$OUTPUT/scientific_comparison_report.json"
./scripts/validate_scientific_outputs.sh --directory "$OUTPUT"
