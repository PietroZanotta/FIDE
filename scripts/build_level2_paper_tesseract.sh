#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ARGS=()
if [[ "${1:-}" == "--no-cache" ]] && tesseract build --help 2>&1 | grep -q -- '--no-cache'; then
  ARGS+=(--no-cache)
elif [[ $# -gt 0 ]]; then
  echo "Usage: ./scripts/build_level2_paper_tesseract.sh [--no-cache]" >&2
  exit 2
fi
command -v tesseract >/dev/null 2>&1 || { echo "tesseract CLI not found; activate .venv first." >&2; exit 2; }
command -v docker >/dev/null 2>&1 || { echo "Docker not found." >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker daemon is not reachable." >&2; exit 2; }
banner "Build paper-facing level-2 invariant correction Tesseract"
tesseract build "${ARGS[@]}" "$ROOT/tesseracts/paper_level2_correction"
docker image inspect mfsi-paper-level2-correction:latest >/dev/null
echo "mfsi-paper-level2-correction:latest"
