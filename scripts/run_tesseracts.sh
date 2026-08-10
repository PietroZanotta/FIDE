#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
BUILD=0
usage(){ cat <<'EOF'
Usage: ./scripts/run_tesseracts.sh [--build]

Checks direct-JAX/Tesseract-kernel parity. If the two Docker images are already
built, also invokes the actual Pasteur/ISI Labs Tesseracts via `tesseract run`.
Use --build to build the images first.
EOF
}
while [[ $# -gt 0 ]]; do case "$1" in
 --build) BUILD=1; shift;; -h|--help) usage; exit 0;;
 *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;; esac; done
[[ "$BUILD" -eq 0 ]] || "$ROOT/scripts/build_tesseracts.sh"
banner "JAX/Tesseract parity + actual Tesseract Core container check"
run_py validate_tesseracts.py
