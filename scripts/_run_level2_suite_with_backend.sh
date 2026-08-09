#!/usr/bin/env bash
# Isolated advanced-level-2 backend helper; existing wrappers are untouched.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
BACKEND="${1:-tesseract}"; shift || true
[[ $# -gt 0 ]] || { echo "internal error: no command supplied" >&2; exit 2; }
case "$BACKEND" in
  jax) export MFSI_BACKEND=jax; exec "$@";;
  tesseract) ;;
  *) echo "backend must be tesseract or jax" >&2; exit 2;;
esac
command -v tesseract >/dev/null 2>&1 || { echo "tesseract CLI not found; activate .venv or use --backend jax." >&2; exit 2; }
command -v docker >/dev/null 2>&1 || { echo "Docker is unavailable; use --backend jax." >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker daemon is not reachable." >&2; exit 2; }
command -v curl >/dev/null 2>&1 || { echo "curl is required." >&2; exit 2; }
if ! docker image inspect mfsi-finite-neural-path:latest >/dev/null 2>&1 || \
   ! docker image inspect mfsi-manybody-neural-path:latest >/dev/null 2>&1; then
  banner "Advanced level-2 images missing; building them"
  "$ROOT/scripts/build_level2_suite_tesseracts.sh"
fi

FINITE_PORT="${MFSI_FINITE_NEURAL_TESSERACT_PORT:-18084}"
MANYBODY_PORT="${MFSI_MANYBODY_TESSERACT_PORT:-18085}"
FINITE_LOG="$(mktemp)"; MANYBODY_LOG="$(mktemp)"
FINITE_PROJECT=""; MANYBODY_PROJECT=""
cleanup(){
  set +e
  [[ -z "$MANYBODY_PROJECT" ]] || tesseract teardown "$MANYBODY_PROJECT" >/dev/null 2>&1
  [[ -z "$FINITE_PROJECT" ]] || tesseract teardown "$FINITE_PROJECT" >/dev/null 2>&1
  rm -f "$FINITE_LOG" "$MANYBODY_LOG"
}
trap cleanup EXIT INT TERM
extract_project_id(){
  "$PY" -c 'import re,sys; s=sys.stdin.read(); m=re.findall(r"[\"'"'"'](?:container_name|project_id)[\"'"'"']\s*:\s*[\"'"'"']([^\"'"'"']+)",s); print(m[-1] if m else "")'
}
banner "Serve advanced level-2 Tesseracts"
tesseract serve -p "$FINITE_PORT" mfsi-finite-neural-path:latest 2>&1 | tee "$FINITE_LOG"
FINITE_PROJECT="$(extract_project_id < "$FINITE_LOG")"
[[ -n "$FINITE_PROJECT" ]] || { echo "Could not determine finite-neural project id." >&2; exit 2; }
tesseract serve -p "$MANYBODY_PORT" mfsi-manybody-neural-path:latest 2>&1 | tee "$MANYBODY_LOG"
MANYBODY_PROJECT="$(extract_project_id < "$MANYBODY_LOG")"
[[ -n "$MANYBODY_PROJECT" ]] || { echo "Could not determine many-body project id." >&2; exit 2; }
export MFSI_BACKEND=tesseract
export MFSI_FINITE_NEURAL_TESSERACT_URL="http://127.0.0.1:${FINITE_PORT}"
export MFSI_MANYBODY_TESSERACT_URL="http://127.0.0.1:${MANYBODY_PORT}"
for URL in "$MFSI_FINITE_NEURAL_TESSERACT_URL" "$MFSI_MANYBODY_TESSERACT_URL"; do
  READY=0
  for _ in $(seq 1 60); do
    if curl -fsS "$URL/health" >/dev/null 2>&1; then READY=1; break; fi
    sleep 1
  done
  [[ "$READY" -eq 1 ]] || { echo "Tesseract failed health check: $URL" >&2; exit 2; }
done
"$@"
