#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
[[ $# -gt 0 ]] || { echo "internal error: no command supplied" >&2; exit 2; }
command -v tesseract >/dev/null 2>&1 || { echo "tesseract CLI not found" >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker daemon is not reachable" >&2; exit 2; }
if ! docker image inspect mfsi-rollout-gradient-engine:latest >/dev/null 2>&1 || \
   ! docker image inspect mfsi-fiber-gradient-engine:latest >/dev/null 2>&1; then
  "$ROOT/scripts/build_gradient_tesseracts.sh"
fi
ROLLOUT_PORT="${MFSI_ROLLOUT_GRADIENT_TESSERACT_PORT:-18087}"
FIBER_PORT="${MFSI_FIBER_GRADIENT_TESSERACT_PORT:-18088}"
ROLLOUT_LOG="$(mktemp)"; FIBER_LOG="$(mktemp)"
ROLLOUT_PROJECT=""; FIBER_PROJECT=""
extract_project_id(){
  "$PY" -c 'import re,sys; s=sys.stdin.read(); m=re.findall(r"[\"'"'"'](?:container_name|project_id)[\"'"'"']\s*:\s*[\"'"'"']([^\"'"'"']+)",s); print(m[-1] if m else "")'
}
cleanup(){
  set +e
  [[ -z "$FIBER_PROJECT" ]] || tesseract teardown "$FIBER_PROJECT" >/dev/null 2>&1
  [[ -z "$ROLLOUT_PROJECT" ]] || tesseract teardown "$ROLLOUT_PROJECT" >/dev/null 2>&1
  rm -f "$ROLLOUT_LOG" "$FIBER_LOG"
}
trap cleanup EXIT INT TERM
banner "Serve differentiable Tesseracts"
tesseract serve -p "$ROLLOUT_PORT" mfsi-rollout-gradient-engine:latest 2>&1 | tee "$ROLLOUT_LOG"
ROLLOUT_PROJECT="$(extract_project_id < "$ROLLOUT_LOG")"
tesseract serve -p "$FIBER_PORT" mfsi-fiber-gradient-engine:latest 2>&1 | tee "$FIBER_LOG"
FIBER_PROJECT="$(extract_project_id < "$FIBER_LOG")"
[[ -n "$ROLLOUT_PROJECT" && -n "$FIBER_PROJECT" ]] || { echo "Could not identify served Tesseracts" >&2; exit 2; }
export MFSI_BACKEND=tesseract
export MFSI_ROLLOUT_GRADIENT_TESSERACT_URL="http://127.0.0.1:${ROLLOUT_PORT}"
export MFSI_FIBER_GRADIENT_TESSERACT_URL="http://127.0.0.1:${FIBER_PORT}"
for URL in "$MFSI_ROLLOUT_GRADIENT_TESSERACT_URL" "$MFSI_FIBER_GRADIENT_TESSERACT_URL"; do
  READY=0
  for _ in $(seq 1 90); do
    if curl -fsS "$URL/health" >/dev/null 2>&1; then READY=1; break; fi
    sleep 1
  done
  [[ "$READY" -eq 1 ]] || { echo "Tesseract failed health check: $URL" >&2; exit 2; }
done
"$@"
