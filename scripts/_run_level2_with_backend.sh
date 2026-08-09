#!/usr/bin/env bash
# Internal level-2 helper. It is intentionally separate from the level-0/1
# two-container wrapper so those established run paths cannot be disturbed.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

BACKEND="${1:-tesseract}"
shift || true
[[ $# -gt 0 ]] || { echo "internal error: no command supplied" >&2; exit 2; }

case "$BACKEND" in
  jax)
    export MFSI_BACKEND=jax
    exec "$@"
    ;;
  tesseract) ;;
  *) echo "backend must be 'tesseract' or 'jax'" >&2; exit 2;;
esac

command -v tesseract >/dev/null 2>&1 || {
  echo "Level-2 Tesseract requested but the 'tesseract' CLI is not installed." >&2
  echo "Run ./scripts/install.sh, or explicitly choose --backend jax." >&2
  exit 2
}
command -v docker >/dev/null 2>&1 || {
  echo "Level-2 Tesseract requested but Docker is unavailable. Choose --backend jax or install Docker." >&2
  exit 2
}
docker info >/dev/null 2>&1 || {
  echo "Docker daemon is not reachable. Start Docker or choose --backend jax." >&2
  exit 2
}
command -v curl >/dev/null 2>&1 || { echo "curl is required for Tesseract REST mode." >&2; exit 2; }

if ! docker image inspect mfsi-fiber-path-adapter:latest >/dev/null 2>&1; then
  banner "Level-2 Tesseract image missing; building it"
  "$ROOT/scripts/build_level2_tesseract.sh"
fi

PORT="${MFSI_LEVEL2_TESSERACT_PORT:-18083}"
LOG_FILE="$(mktemp)"
PROJECT_ID=""
cleanup(){
  set +e
  [[ -z "$PROJECT_ID" ]] || tesseract teardown "$PROJECT_ID" >/dev/null 2>&1
  rm -f "$LOG_FILE"
}
trap cleanup EXIT INT TERM

extract_project_id(){
  "$PY" -c 'import re,sys; s=sys.stdin.read(); m=re.findall(r"[\"'"'"'](?:container_name|project_id)[\"'"'"']\s*:\s*[\"'"'"']([^\"'"'"']+)",s); print(m[-1] if m else "")'
}

banner "Serve level-2 FiberPathAdapter Tesseract"
tesseract serve -p "$PORT" mfsi-fiber-path-adapter:latest 2>&1 | tee "$LOG_FILE"
PROJECT_ID="$(extract_project_id < "$LOG_FILE")"
[[ -n "$PROJECT_ID" ]] || { echo "Could not determine level-2 Tesseract project id." >&2; exit 2; }

export MFSI_BACKEND=tesseract
export MFSI_LEVEL2_TESSERACT_URL="http://127.0.0.1:${PORT}"
READY=0
for _ in $(seq 1 60); do
  if curl -fsS "$MFSI_LEVEL2_TESSERACT_URL/health" >/dev/null 2>&1; then READY=1; break; fi
  sleep 1
done
[[ "$READY" -eq 1 ]] || { echo "Level-2 Tesseract failed health check." >&2; exit 2; }

"$@"
