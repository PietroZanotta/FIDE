#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
BACKEND="${1:-tesseract}"; shift || true
[[ $# -gt 0 ]] || { echo "internal error: no command supplied" >&2; exit 2; }
if [[ "$BACKEND" == "jax" ]]; then
  export MFSI_BACKEND=jax
  exec "$@"
fi
[[ "$BACKEND" == "tesseract" ]] || { echo "backend must be tesseract or jax" >&2; exit 2; }
command -v tesseract >/dev/null 2>&1 || { echo "tesseract CLI not found; activate .venv or use --backend jax." >&2; exit 2; }
command -v docker >/dev/null 2>&1 || { echo "Docker is unavailable; use --backend jax." >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker daemon is not reachable." >&2; exit 2; }
if ! docker image inspect mfsi-paper-level2-correction:latest >/dev/null 2>&1; then
  "$ROOT/scripts/build_level2_paper_tesseract.sh"
fi
PORT="${MFSI_PAPER_LEVEL2_TESSERACT_PORT:-18086}"
LOG="$(mktemp)"; PROJECT=""
cleanup(){
  set +e
  [[ -z "$PROJECT" ]] || tesseract teardown "$PROJECT" >/dev/null 2>&1
  rm -f "$LOG"
}
trap cleanup EXIT INT TERM
tesseract serve -p "$PORT" mfsi-paper-level2-correction:latest 2>&1 | tee "$LOG"
PROJECT="$($PY -c 'import re,sys; s=sys.stdin.read(); m=re.findall(r"[\"'"'"'](?:container_name|project_id)[\"'"'"']\s*:\s*[\"'"'"']([^\"'"'"']+)",s); print(m[-1] if m else "")' < "$LOG")"
[[ -n "$PROJECT" ]] || { echo "Could not determine Tesseract project id." >&2; exit 2; }
export MFSI_BACKEND=tesseract
export MFSI_PAPER_LEVEL2_TESSERACT_URL="http://127.0.0.1:${PORT}"
READY=0
for _ in $(seq 1 60); do
  if curl -fsS "$MFSI_PAPER_LEVEL2_TESSERACT_URL/health" >/dev/null 2>&1; then READY=1; break; fi
  sleep 1
done
[[ "$READY" -eq 1 ]] || { echo "Tesseract failed health check." >&2; exit 2; }
"$@"
