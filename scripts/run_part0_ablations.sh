#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
QUICK=0; FORCE_GRAD=0
while [[ $# -gt 0 ]]; do case "$1" in
  --quick) QUICK=1; shift;;
  --force-gradient) FORCE_GRAD=1; shift;;
  -h|--help) echo "Usage: ./scripts/run_part0_ablations.sh [--quick] [--force-gradient]"; exit 0;;
  *) echo "Unknown argument: $1" >&2; exit 2;;
esac; done

if [[ "$FORCE_GRAD" -eq 1 || ! -f "$ROOT/results/ablation_metrics.json" ]]; then
  banner "Gradient / implicit-differentiation ablation"
  run_py ablate_and_benchmark.py
else
  banner "Gradient ablation: reuse results/ablation_metrics.json (pass --force-gradient to rerun)"
fi

if [[ "$QUICK" -eq 1 ]]; then
  banner "Part-0 prescribed ablations (quick/debug)"
  run_py part0_ablations.py --quick
else
  # Separate processes keep JAX compilation/memory bounded and make long paper
  # ablations easier to resume/debug. Each section merges into the same JSON.
  for SECTION in capacity batch geometry safety differentiation; do
    banner "Part-0 prescribed ablation: $SECTION"
    run_py part0_ablations.py --section "$SECTION"
  done
fi
