#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! python -c 'import tesseract_core' >/dev/null 2>&1; then
  echo "Tesseract Core is not importable; activate .venv first." >&2
  exit 1
fi

TESSERACT_BIN="$(command -v tesseract || true)"
if [[ -z "${TESSERACT_BIN}" ]] || ! "${TESSERACT_BIN}" build --help 2>&1 | grep -q "Build a new Tesseract"; then
  echo "The Pasteur Labs Tesseract Core CLI is not active." >&2
  exit 1
fi

"${TESSERACT_BIN}" build tesseracts/scientific_relaxation
"${TESSERACT_BIN}" build tesseracts/scientific_projection
