#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NO_CACHE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cache) NO_CACHE=1; shift;;
    -h|--help) echo "Usage: ./scripts/build_gradient_tesseracts.sh [--no-cache]"; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
command -v tesseract >/dev/null 2>&1 || { echo "tesseract CLI not found" >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker daemon is not reachable" >&2; exit 2; }
ARGS=()
if [[ "$NO_CACHE" -eq 1 ]] && tesseract build --help 2>&1 | rg -q -- '--no-cache'; then
  ARGS+=(--no-cache)
fi
for PROJECT in rollout_gradient_engine fiber_gradient_engine; do
  banner "Build differentiable Tesseract: $PROJECT"
  tesseract build "${ARGS[@]}" "$ROOT/tesseracts/$PROJECT"
done
docker image inspect mfsi-rollout-gradient-engine:latest >/dev/null
docker image inspect mfsi-fiber-gradient-engine:latest >/dev/null
