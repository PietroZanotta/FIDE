#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/_common.sh"

TRAIN_SEEDS_STR="${TRAIN_SEEDS:-20260808 20260818 20260828}"
EVAL_SEEDS_STR="${EVAL_SEEDS:-20260821 20260822 20260823 20260824}"
BACKEND="${MFSI_BACKEND:-tesseract}"
OUT_DIR="${SWEEP_OUT:-}"
MODE="full"
FORCE=0

usage() {
  cat <<'EOF_USAGE'
Usage: ./scripts/run_multiseed_b.sh [options]

  --backend tesseract|jax      evaluation backend (default: tesseract)
  --train-seeds "S1 S2 ..."   independent training seeds
  --eval-seeds  "S1 S2 ..."   independent evaluation seeds
  --quick                      shorter debug budgets; not paper metrics
  --smoke                      minimal plumbing-only budgets
  --out DIR                    output directory
  --force                      recompute completed train/eval pairs
  -h, --help                   show this help

By default an interrupted sweep resumes from existing checkpoints/results.
If --out is omitted, outputs are backend-specific:

  results/multiseed/example_b/<backend>/

Environment equivalents:
  MFSI_BACKEND, TRAIN_SEEDS, EVAL_SEEDS, SWEEP_OUT
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      [[ $# -ge 2 ]] || { echo "--backend requires a value" >&2; exit 2; }
      BACKEND="$2"; shift 2 ;;
    --train-seeds)
      [[ $# -ge 2 ]] || { echo "--train-seeds requires a value" >&2; exit 2; }
      TRAIN_SEEDS_STR="$2"; shift 2 ;;
    --eval-seeds)
      [[ $# -ge 2 ]] || { echo "--eval-seeds requires a value" >&2; exit 2; }
      EVAL_SEEDS_STR="$2"; shift 2 ;;
    --quick)
      MODE="quick"; shift ;;
    --smoke)
      MODE="smoke"; shift ;;
    --out)
      [[ $# -ge 2 ]] || { echo "--out requires a directory" >&2; exit 2; }
      OUT_DIR="$2"; shift 2 ;;
    --force)
      FORCE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

[[ "$BACKEND" == "tesseract" || "$BACKEND" == "jax" ]] || {
  echo "backend must be 'tesseract' or 'jax'" >&2
  exit 2
}

[[ -n "${TRAIN_SEEDS_STR//[[:space:],]/}" ]] || {
  echo "training seed list cannot be empty" >&2
  exit 2
}
[[ -n "${EVAL_SEEDS_STR//[[:space:],]/}" ]] || {
  echo "evaluation seed list cannot be empty" >&2
  exit 2
}

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$ROOT/results/multiseed/example_b/$BACKEND"
fi

ARGS=(
  --backend "$BACKEND"
  --train-seeds "$TRAIN_SEEDS_STR"
  --eval-seeds "$EVAL_SEEDS_STR"
  --out "$OUT_DIR"
)

[[ "$MODE" == "full" ]] || ARGS+=(--"$MODE")
[[ "$FORCE" -eq 0 ]] || ARGS+=(--force)

banner "Experiment B multi-training / multi-evaluation sweep ($MODE, backend: $BACKEND)"
echo "Training seeds : $TRAIN_SEEDS_STR"
echo "Evaluation seeds: $EVAL_SEEDS_STR"
echo "Output directory: $OUT_DIR"

"$ROOT/scripts/_run_with_backend.sh" \
  "$BACKEND" \
  "$PY" \
  sweep_example_b.py \
  "${ARGS[@]}"

echo
echo "Sweep outputs: $OUT_DIR"
echo "Authoritative aggregate: $OUT_DIR/aggregate.json"
