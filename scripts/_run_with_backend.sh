#!/usr/bin/env bash
# Internal helper: execute a command with either direct JAX or two served
# Pasteur/ISI Labs Tesseract Core containers.
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
  echo "Tesseract backend requested but the Pasteur/ISI Labs 'tesseract' CLI is not installed." >&2
  echo "Run ./scripts/install.sh, or explicitly choose --backend jax." >&2
  exit 2
}
command -v docker >/dev/null 2>&1 || {
  echo "Tesseract backend requested but Docker is unavailable. Choose --backend jax or install Docker." >&2
  exit 2
}
docker info >/dev/null 2>&1 || {
  echo "Docker daemon is not reachable. Start Docker or choose --backend jax." >&2
  exit 2
}
command -v curl >/dev/null 2>&1 || {
  echo "curl is required for the REST-backed Tesseract mode." >&2; exit 2
}

if ! docker image inspect mfsi-reference-transport:latest >/dev/null 2>&1 || \
   ! docker image inspect mfsi-moment-fiber-realizer:latest >/dev/null 2>&1; then
  banner "Tesseract images missing; building them"
  "$ROOT/scripts/build_tesseracts.sh"
fi

REF_PORT="${MFSI_REFERENCE_TESSERACT_PORT:-18081}"
FIB_PORT="${MFSI_FIBER_TESSERACT_PORT:-18082}"

extract_project_id(){
  # Tesseract Core 1.11 calls this identifier ``container_name``; older
  # releases returned ``project_id``. Accept both output formats.
  "$PY" -c 'import re,sys; s=sys.stdin.read(); m=re.findall(r"[\"'"'"'](?:container_name|project_id)[\"'"'"']\s*:\s*[\"'"'"']([^\"'"'"']+)",s); print(m[-1] if m else "")'
}

banner "Serve Tesseract backend"
REF_LOG="$(mktemp)"; FIB_LOG="$(mktemp)"
REF_PROJECT=""; FIB_PROJECT=""
cleanup(){
  set +e
  [[ -z "$FIB_PROJECT" ]] || tesseract teardown "$FIB_PROJECT" >/dev/null 2>&1
  [[ -z "$REF_PROJECT" ]] || tesseract teardown "$REF_PROJECT" >/dev/null 2>&1
  rm -f "$REF_LOG" "$FIB_LOG"
}
trap cleanup EXIT INT TERM

# `tesseract serve` launches the container and returns project metadata.
tesseract serve -p "$REF_PORT" mfsi-reference-transport:latest 2>&1 | tee "$REF_LOG"
REF_PROJECT="$(extract_project_id < "$REF_LOG")"
[[ -n "$REF_PROJECT" ]] || { echo "Could not determine ReferenceTransport Tesseract project id." >&2; exit 2; }

tesseract serve -p "$FIB_PORT" mfsi-moment-fiber-realizer:latest 2>&1 | tee "$FIB_LOG"
FIB_PROJECT="$(extract_project_id < "$FIB_LOG")"
[[ -n "$FIB_PROJECT" ]] || { echo "Could not determine MomentFiberRealizer Tesseract project id." >&2; exit 2; }

export MFSI_BACKEND=tesseract
export MFSI_REFERENCE_TESSERACT_URL="http://127.0.0.1:${REF_PORT}"
export MFSI_FIBER_TESSERACT_URL="http://127.0.0.1:${FIB_PORT}"

for URL in "$MFSI_REFERENCE_TESSERACT_URL" "$MFSI_FIBER_TESSERACT_URL"; do
  ok=0
  for _ in $(seq 1 60); do
    if curl -fsS "$URL/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 1
  done
  [[ "$ok" -eq 1 ]] || { echo "Tesseract server failed health check: $URL" >&2; exit 2; }
done

"$@"
