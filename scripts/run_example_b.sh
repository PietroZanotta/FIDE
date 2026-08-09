#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ARGS=(); BACKEND="${MFSI_BACKEND:-tesseract}"
usage(){ cat <<'EOF'
Usage: ./scripts/run_example_b.sh [--backend tesseract|jax] [--retrain] [--quick] [--no-plots] [--seed N]

Default backend: tesseract. Neural-network training remains native JAX; the
ReferenceTransport and MomentFiberRealizer evaluations in generation/evaluation
use the selected backend.
EOF
}
while [[ $# -gt 0 ]]; do case "$1" in
 --backend) BACKEND="$2"; shift 2;;
 --retrain|--quick|--no-plots) ARGS+=("$1"); shift;;
 --seed) ARGS+=(--seed "$2"); shift 2;; -h|--help) usage; exit 0;;
 *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;; esac; done
[[ "$BACKEND" == tesseract || "$BACKEND" == jax ]] || { echo "backend must be tesseract or jax" >&2; exit 2; }
banner "Experiment B (backend: $BACKEND)"
"$ROOT/scripts/_run_with_backend.sh" "$BACKEND" "$PY" example_b.py --backend "$BACKEND" "${ARGS[@]}"
echo "Outputs: $ROOT/results/example_b/"
