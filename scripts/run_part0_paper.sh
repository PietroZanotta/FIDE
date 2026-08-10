#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
BACKEND="${MFSI_BACKEND:-tesseract}"; QUICK=0; RETRAIN=0; MULTISEED=0; REBUILD=0
TRAIN_SEEDS="20260808 20260818 20260828"; EVAL_SEEDS="20260821 20260822 20260823 20260824"
usage(){ cat <<'EOF'
Usage: ./scripts/run_part0_paper.sh [options]
  --backend tesseract|jax     experiment backend (default: tesseract)
  --quick                     quick B + quick ablations (debug, not paper metrics)
  --retrain                   retrain A and B before evaluation
  --multiseed                 run B training-seed x evaluation-seed sweep
  --train-seeds "..."         seeds for --multiseed
  --eval-seeds "..."          seeds for --multiseed
  --rebuild-tesseracts        rebuild images before systems benchmark

Runs Experiment A, Experiment B, prescribed Part-0 ablations, the Tesseract
systems benchmark, and finally prints the consolidated terminal tables.
EOF
}
while [[ $# -gt 0 ]]; do case "$1" in
 --backend) BACKEND="$2"; shift 2;; --quick) QUICK=1; shift;; --retrain) RETRAIN=1; shift;;
 --multiseed) MULTISEED=1; shift;; --train-seeds) TRAIN_SEEDS="$2"; shift 2;; --eval-seeds) EVAL_SEEDS="$2"; shift 2;;
 --rebuild-tesseracts) REBUILD=1; shift;; -h|--help) usage; exit 0;; *) usage >&2; exit 2;; esac; done
A=(--backend "$BACKEND"); B=(--backend "$BACKEND")
[[ "$RETRAIN" -eq 0 ]] || { A+=(--retrain); B+=(--retrain); }
[[ "$QUICK" -eq 0 ]] || B+=(--quick)
"$ROOT/scripts/run_example_a.sh" "${A[@]}"
"$ROOT/scripts/run_example_b.sh" "${B[@]}"
"$ROOT/scripts/run_mgd_validation.sh"
"$ROOT/scripts/run_tesseracts.sh"
ABL=(); [[ "$QUICK" -eq 0 ]] || ABL+=(--quick)
"$ROOT/scripts/run_part0_ablations.sh" "${ABL[@]}"
SYS=(); [[ "$REBUILD" -eq 0 ]] || SYS+=(--rebuild)
"$ROOT/scripts/run_tesseract_systems.sh" "${SYS[@]}"
if [[ "$MULTISEED" -eq 1 ]]; then
  SW=(--backend "$BACKEND" --train-seeds "$TRAIN_SEEDS" --eval-seeds "$EVAL_SEEDS")
  [[ "$QUICK" -eq 0 ]] || SW+=(--quick)
  "$ROOT/scripts/run_multiseed_b.sh" "${SW[@]}"
fi
banner "Part-0 consolidated tables"
run_py "$ROOT/report_results.py" --results-root "$ROOT/results" --backend "$BACKEND"
