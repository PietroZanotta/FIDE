#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

BACKEND="${MFSI_BACKEND:-tesseract}"
ARGS=()
usage(){ cat <<'EOF'
Usage: ./scripts/run_level2.sh [--backend tesseract|jax] [--quick] [--no-plots]

Run the isolated level-2 fiber-adapted schedule experiment. Tesseract is the
default backend; --backend jax runs the identical JAX recipe in-process.
Outputs are separated under results/level2_schedule/<backend>/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --quick|--no-plots) ARGS+=("$1"); shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ "$BACKEND" == tesseract || "$BACKEND" == jax ]] || {
  echo "backend must be tesseract or jax" >&2; exit 2;
}

banner "Level 2: fiber-adapted reference schedule (backend: $BACKEND)"
"$ROOT/scripts/_run_level2_with_backend.sh" "$BACKEND" \
  "$PY" "$ROOT/level2_schedule.py" --backend "$BACKEND" "${ARGS[@]}"
