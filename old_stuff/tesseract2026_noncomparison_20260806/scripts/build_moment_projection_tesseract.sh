#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! python -c 'import tesseract_core' >/dev/null 2>&1; then
  cat >&2 <<'MSG'
Tesseract Core is not importable in the active Python environment.
Activate the repository virtual environment first:

  source .venv/bin/activate
MSG
  exit 1
fi

TESSERACT_BIN="$(command -v tesseract || true)"
if [[ -z "${TESSERACT_BIN}" ]]; then
  echo "The Tesseract Core CLI is not on PATH. Activate .venv first." >&2
  exit 1
fi

TESSERACT_HELP="$("${TESSERACT_BIN}" build --help 2>&1 || true)"
if [[ "${TESSERACT_HELP}" != *"Build a new Tesseract"* ]]; then
  cat >&2 <<MSG
'${TESSERACT_BIN}' is not the Pasteur Labs Tesseract Core CLI.
It is commonly the unrelated OCR executable. Activate the project environment:

  source .venv/bin/activate

Then confirm:

  python -c 'import tesseract_core; print(tesseract_core.__version__)'
  command -v tesseract
MSG
  exit 1
fi

exec "${TESSERACT_BIN}" build tesseracts/moment_projection
