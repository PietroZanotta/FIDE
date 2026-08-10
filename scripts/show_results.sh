set -euo pipefail
source "$(dirname "$0")/_common.sh"
BACKEND="${MFSI_BACKEND:-tesseract}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --no-multiseed) ARGS+=(--no-multiseed); shift;;
    -h|--help)
      echo "Usage: ./scripts/show_results.sh [--backend tesseract|jax] [--no-multiseed]"
      exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
run_py "$ROOT/report_results.py" --results-root "$ROOT/results" --backend "$BACKEND" "${ARGS[@]}"
