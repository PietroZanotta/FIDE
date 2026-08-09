#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

usage(){ cat <<'EOF'
Usage: ./scripts/build_level2_suite_tesseracts.sh [--no-cache]

Build the two isolated advanced level-2 images:
  mfsi-finite-neural-path:latest
  mfsi-manybody-neural-path:latest

This does not rebuild the established level-0/1 or scalar level-2 images.
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
command -v tesseract >/dev/null 2>&1 || { echo "tesseract CLI not found; activate .venv first." >&2; exit 2; }
command -v docker >/dev/null 2>&1 || { echo "Docker not found." >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker daemon is not reachable." >&2; exit 2; }
ARGS=()
if [[ "$NO_CACHE" -eq 1 ]] && tesseract build --help 2>&1 | grep -q -- '--no-cache'; then ARGS+=(--no-cache); fi
banner "Build advanced level-2 Tesseract: FiniteNeuralPath"
tesseract build "${ARGS[@]}" "$ROOT/tesseracts/finite_neural_path"
banner "Build advanced level-2 Tesseract: ManybodyNeuralPath"
tesseract build "${ARGS[@]}" "$ROOT/tesseracts/manybody_neural_path"
docker image inspect mfsi-finite-neural-path:latest >/dev/null
docker image inspect mfsi-manybody-neural-path:latest >/dev/null
echo "mfsi-finite-neural-path:latest"
echo "mfsi-manybody-neural-path:latest"
