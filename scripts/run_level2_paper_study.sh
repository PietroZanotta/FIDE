#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
BACKEND="${MFSI_BACKEND:-tesseract}"
ARGS=()
usage(){
  cat <<'EOF'
Usage: ./scripts/run_level2_paper_study.sh [--backend tesseract|jax]
       [--quick] [--seeds "401 402 403 404 405"] [--no-plots]

Run the isolated, paper-facing N=32 level-2 study. Standard mode uses five
independent finite particle banks and reports seed-level 95% intervals.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --quick|--no-plots|--aggregate-existing) ARGS+=("$1"); shift;;
    --seeds) ARGS+=("$1" "$2"); shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ "$BACKEND" == "jax" || "$BACKEND" == "tesseract" ]] || { echo "invalid backend" >&2; exit 2; }
banner "Paper-facing level 2 study (backend: $BACKEND)"
"$ROOT/scripts/_run_level2_paper_with_backend.sh" "$BACKEND" \
  "$PY" "$ROOT/level2_paper_study.py" --backend "$BACKEND" "${ARGS[@]}"
