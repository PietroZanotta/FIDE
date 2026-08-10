#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
FROM="a_b"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2;;
    -h|--help)
      echo "Usage: ./scripts/run_paper_jax_regression.sh [--from a_b|part0|level2|stages|validate]"
      exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
case "$FROM" in
  a_b) START=0;; part0) START=1;; level2) START=2;; stages) START=3;; validate) START=4;;
  *) echo "invalid --from phase: $FROM" >&2; exit 2;;
esac

if [[ "$START" -le 0 ]]; then
  banner "Paper regression: Experiments A and B (JAX)"
  "$ROOT/scripts/run_example_a.sh" --backend jax
  "$ROOT/scripts/run_example_b.sh" --backend jax
fi
if [[ "$START" -le 1 ]]; then
  banner "Paper regression: remaining Part 0 (JAX)"
  "$ROOT/scripts/run_mgd_validation.sh"
  "$ROOT/scripts/run_part0_ablations.sh"
  "$ROOT/scripts/run_multiseed_b.sh" \
    --backend jax \
    --train-seeds "101 102 103 104 105 106 107 108 109 110" \
    --eval-seeds "201 202 203 204 205 206 207 208 209 210"
fi

if [[ "$START" -le 2 ]]; then
  banner "Paper regression: Level 2 (JAX)"
  "$ROOT/scripts/run_level2.sh" --backend jax
  "$ROOT/scripts/run_level2_suite.sh" --backend jax
  "$ROOT/scripts/run_level2_paper_study.sh" --backend jax
fi

if [[ "$START" -le 3 ]]; then
  banner "Paper regression: frozen Stage 3/4 drivers (JAX)"
  run_py "$ROOT/stage3_rollout_adaptation.py"
  "$ROOT/scripts/prepare_stage3b_base_models.sh"
  run_py "$ROOT/stage3b_confirmatory.py"
  run_py "$ROOT/stage4_fiber_design.py"
  run_py "$ROOT/stage4b_fiber_design_confirmatory.py"
fi

if [[ "$START" -le 4 ]]; then
  banner "Paper regression: artifact validators"
  run_py "$ROOT/validate_stage3_rollout_adaptation.py"
  run_py "$ROOT/validate_stage3b_confirmatory.py"
  run_py "$ROOT/validate_stage4_fiber_design.py"
  run_py "$ROOT/validate_stage4b_fiber_design.py"
  run_py "$ROOT/validate_tesseracts.py"
fi

banner "Paper JAX regression complete"
