#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
OUTPUT="artifacts/scientific_comparison_micro"
rm -rf "$OUTPUT"
python -m manybody_completion.comparison_cli \
  --config configs/scientific_comparison_micro.yaml \
  --output "$OUTPUT" \
  --rerun-flow
python scripts/run_composed_gradient_probe.py \
  --output "$OUTPUT/composed_gradient_probe.json"
python scripts/build_comparison_report.py \
  --report "$OUTPUT/scientific_comparison_report.json" \
  --output "$OUTPUT/SCIENTIFIC_COMPARISON_SUMMARY.md"
python scripts/validate_scientific_outputs.py --directory "$OUTPUT"
