#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

usage(){ cat <<'EOF'
Usage: ./scripts/build_tesseracts.sh [--no-cache]

Build the two Pasteur/ISI Labs Tesseract Core projects into Docker images using
`tesseract build`:
  mfsi-reference-transport:latest
  mfsi-moment-fiber-realizer:latest

Requires the `tesseract` CLI (from tesseract-core) and a running Docker engine.
EOF
}
NO_CACHE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cache) NO_CACHE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

command -v tesseract >/dev/null 2>&1 || {
  echo "tesseract CLI not found. Run ./scripts/install.sh first." >&2; exit 2;
}
command -v docker >/dev/null 2>&1 || {
  echo "Docker not found. Install/start Docker before building Tesseracts." >&2; exit 2;
}
docker info >/dev/null 2>&1 || {
  echo "Docker is installed but the daemon is not reachable." >&2; exit 2;
}

banner "Build Tesseract 1: ReferenceTransport"
ARGS=()
# Keep the public wrapper stable even if a future Tesseract Core changes its
# cache flags. For now --no-cache is forwarded only when supported by help.
if [[ "$NO_CACHE" -eq 1 ]] && tesseract build --help 2>&1 | grep -q -- '--no-cache'; then
  ARGS+=(--no-cache)
fi
tesseract build "${ARGS[@]}" "$ROOT/tesseracts/reference_transport"

banner "Build Tesseract 2: MomentFiberRealizer"
tesseract build "${ARGS[@]}" "$ROOT/tesseracts/moment_fiber_realizer"

banner "Built images"
docker image inspect mfsi-reference-transport:latest >/dev/null
docker image inspect mfsi-moment-fiber-realizer:latest >/dev/null
echo "mfsi-reference-transport:latest"
echo "mfsi-moment-fiber-realizer:latest"
