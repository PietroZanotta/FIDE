#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
BACKEND="${MFSI_BACKEND:-tesseract}"
EXPERIMENT=both
ARGS=()
usage(){ cat <<'EOF'
Usage: ./scripts/run_level2_suite.sh [--backend tesseract|jax]
       [--experiment finite_neural|manybody|both] [--quick] [--no-plots]

Run the advanced two-experiment level-2 suite. Tesseract is the default.
Outputs are isolated under results/level2_suite/<experiment>/<backend>/.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --experiment) EXPERIMENT="$2"; shift 2;;
    --quick|--no-plots) ARGS+=("$1"); shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ "$BACKEND" == tesseract || "$BACKEND" == jax ]] || { echo "invalid backend" >&2; exit 2; }
[[ "$EXPERIMENT" == finite_neural || "$EXPERIMENT" == manybody || "$EXPERIMENT" == both ]] || { echo "invalid experiment" >&2; exit 2; }
banner "Advanced level 2 suite ($EXPERIMENT, backend: $BACKEND)"
"$ROOT/scripts/_run_level2_suite_with_backend.sh" "$BACKEND" \
  "$PY" "$ROOT/level2_suite.py" --backend "$BACKEND" --experiment "$EXPERIMENT" "${ARGS[@]}"
