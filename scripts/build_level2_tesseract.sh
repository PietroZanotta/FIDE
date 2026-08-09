#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

usage(){ cat <<'EOF'
Usage: ./scripts/build_level2_tesseract.sh [--no-cache]

Build only the isolated level-2 FiberPathAdapter Tesseract image:
  mfsi-fiber-path-adapter:latest

The two level-0/1 images and runners are not rebuilt or modified.
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
  echo "Docker not found. Install/start Docker before building the level-2 Tesseract." >&2; exit 2;
}
docker info >/dev/null 2>&1 || {
  echo "Docker is installed but the daemon is not reachable." >&2; exit 2;
}

ARGS=()
if [[ "$NO_CACHE" -eq 1 ]] && tesseract build --help 2>&1 | grep -q -- '--no-cache'; then
  ARGS+=(--no-cache)
fi
banner "Build level-2 Tesseract: FiberPathAdapter"
tesseract build "${ARGS[@]}" "$ROOT/tesseracts/fiber_path_adapter"
docker image inspect mfsi-fiber-path-adapter:latest >/dev/null
echo "mfsi-fiber-path-adapter:latest"
