#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
MODE=load; VALIDATE_ONLY=0; BACKEND="${MFSI_BACKEND:-tesseract}"
usage(){ cat <<'EOF'
Usage: ./scripts/run_example_a.sh [--backend tesseract|jax] [--retrain | --refine] [--validate-only]

Default backend: tesseract (Pasteur/ISI Labs Tesseract Core containers).
Training/holdout validation is native JAX; the learned scientific components in
the matched generation benchmark use the selected backend.
EOF
}
while [[ $# -gt 0 ]]; do case "$1" in
 --backend) BACKEND="$2"; shift 2;;
 --retrain) MODE=retrain; shift;; --refine) MODE=refine; shift;;
 --validate-only) VALIDATE_ONLY=1; shift;; -h|--help) usage; exit 0;;
 *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;; esac; done
[[ "$BACKEND" == tesseract || "$BACKEND" == jax ]] || { echo "backend must be tesseract or jax" >&2; exit 2; }

banner "Experiment A: learned-pipeline validation (training/validation: JAX)"
case "$MODE" in retrain) run_py validate_pipeline.py --retrain;; refine) run_py validate_pipeline.py --refine;; *) run_py validate_pipeline.py;; esac
if [[ "$VALIDATE_ONLY" -eq 0 ]]; then
  banner "Experiment A: matched benchmark (backend: $BACKEND)"
  "$ROOT/scripts/_run_with_backend.sh" "$BACKEND" "$PY" benchmark_methods.py --backend "$BACKEND"
fi
echo "Outputs: $ROOT/results/"
