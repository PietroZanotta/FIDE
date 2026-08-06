#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SEED=20260805
OUTPUT=""
RUN_BACKEND_PARITY=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --skip-backend-parity)
      RUN_BACKEND_PARITY=false
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--seed N] [--output DIRECTORY] [--skip-backend-parity]" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="artifacts/scientific_comparison_full_seed_${SEED}"
fi

python -m manybody_completion.comparison_cli \
  --config configs/scientific_comparison_full.yaml \
  --output "$OUTPUT" \
  --seed "$SEED" \
  --rerun-flow

python scripts/run_composed_gradient_probe.py \
  --experiment-report "$OUTPUT/scientific_comparison_report.json" \
  --output "$OUTPUT/composed_gradient_probe.json"
python scripts/build_comparison_report.py \
  --report "$OUTPUT/scientific_comparison_report.json" \
  --output "$OUTPUT/SCIENTIFIC_COMPARISON_SUMMARY.md"
python scripts/validate_scientific_outputs.py --directory "$OUTPUT"

if [[ "$RUN_BACKEND_PARITY" == true ]]; then
  ./scripts/run_backend_pipeline_parity.sh \
    --seed "$SEED" \
    --output "$OUTPUT/backend_pipeline_parity"
fi

echo "Single-seed acceptance run passed: $OUTPUT"
