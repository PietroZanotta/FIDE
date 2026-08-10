set -euo pipefail
source "$(dirname "$0")/_common.sh"

BACKEND="${MFSI_BACKEND:-tesseract}"
A_MODE="load"
B_RETRAIN=0
B_QUICK=0
B_NO_PLOTS=0
B_SEED=""
REPORT_ONLY=0
NO_MULTISEED=0

usage() {
  cat <<'EOF'
Usage: ./scripts/run_experiments_and_report.sh [options]

Runs Experiment A and Experiment B, then prints terminal tables containing the
main benchmark results and validation diagnostics.

Options:
  --backend tesseract|jax  Execution backend for experiment generation/evaluation
                           (default: tesseract)
  --retrain                Retrain both learned experiments
  --retrain-a              Retrain Experiment A only
  --refine-a               Continue/refine Experiment A instead of retraining
  --retrain-b              Retrain Experiment B only
  --quick-b                Use Experiment B's quick/debug budget
  --no-plots               Do not regenerate Experiment B plots
  --seed N                 Experiment B training/evaluation seed
  --report-only            Do not run experiments; report existing results/
  --no-multiseed           Omit any available multi-seed aggregate table
  -h, --help               Show this help

Examples:
  ./scripts/run_experiments_and_report.sh
  ./scripts/run_experiments_and_report.sh --backend jax
  ./scripts/run_experiments_and_report.sh --backend jax --quick-b
  ./scripts/run_experiments_and_report.sh --report-only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --retrain) A_MODE="retrain"; B_RETRAIN=1; shift;;
    --retrain-a) A_MODE="retrain"; shift;;
    --refine-a) A_MODE="refine"; shift;;
    --retrain-b) B_RETRAIN=1; shift;;
    --quick-b) B_QUICK=1; shift;;
    --no-plots) B_NO_PLOTS=1; shift;;
    --seed) B_SEED="$2"; shift 2;;
    --report-only) REPORT_ONLY=1; shift;;
    --no-multiseed) NO_MULTISEED=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

[[ "$BACKEND" == "tesseract" || "$BACKEND" == "jax" ]] || {
  echo "backend must be tesseract or jax" >&2
  exit 2
}

if [[ "$REPORT_ONLY" -eq 0 ]]; then
  A_ARGS=(--backend "$BACKEND")
  case "$A_MODE" in
    retrain) A_ARGS+=(--retrain);;
    refine) A_ARGS+=(--refine);;
  esac
  "$ROOT/scripts/run_example_a.sh" "${A_ARGS[@]}"

  B_ARGS=(--backend "$BACKEND")
  [[ "$B_RETRAIN" -eq 1 ]] && B_ARGS+=(--retrain)
  [[ "$B_QUICK" -eq 1 ]] && B_ARGS+=(--quick)
  [[ "$B_NO_PLOTS" -eq 1 ]] && B_ARGS+=(--no-plots)
  [[ -n "$B_SEED" ]] && B_ARGS+=(--seed "$B_SEED")
  "$ROOT/scripts/run_example_b.sh" "${B_ARGS[@]}"
fi

banner "Combined Experiment A + B results"
REPORT_ARGS=(--results-root "$ROOT/results" --backend "$BACKEND")
[[ "$NO_MULTISEED" -eq 1 ]] && REPORT_ARGS+=(--no-multiseed)
run_py "$ROOT/report_results.py" "${REPORT_ARGS[@]}"
