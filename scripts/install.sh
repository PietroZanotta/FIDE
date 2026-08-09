#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
WITH_TESSERACT=1; PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
usage(){ cat <<USAGE
Usage: ./scripts/install.sh [--jax-only] [--python VERSION_OR_PATH]
Creates .venv and installs dependencies. Default Python: 3.12.
--jax-only skips Tesseract Core. Docker is separately required for container builds.
USAGE
}
while [[ $# -gt 0 ]]; do case "$1" in
  --jax-only) WITH_TESSERACT=0; shift;;
  --python) PYTHON_VERSION="$2"; shift 2;;
  -h|--help) usage; exit 0;;
  *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;; esac; done
rm -rf .venv
if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYTHON_VERSION" .venv
  uv pip install --python .venv/bin/python -r requirements.txt
  [[ "$WITH_TESSERACT" -eq 0 ]] || uv pip install --python .venv/bin/python -r requirements-tesseract.txt
else
  if [[ -x "$PYTHON_VERSION" ]]; then PYBIN="$PYTHON_VERSION";
  elif command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then PYBIN="$(command -v "python${PYTHON_VERSION}")";
  else PYBIN="$(command -v python3 || true)"; fi
  [[ -n "$PYBIN" ]] || { echo "Install Python $PYTHON_VERSION or uv." >&2; exit 2; }
  "$PYBIN" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  [[ "$WITH_TESSERACT" -eq 0 ]] || .venv/bin/python -m pip install -r requirements-tesseract.txt
fi
.venv/bin/python - <<'PY'
import jax, numpy, scipy, matplotlib
print("JAX", jax.__version__, "devices", jax.devices())
print("NumPy", numpy.__version__, "SciPy", scipy.__version__, "Matplotlib", matplotlib.__version__)
try: import tesseract_core; print("Tesseract Core installed")
except Exception as exc: print("Tesseract Core unavailable:", exc)
PY
echo "Ready. Activate with: source .venv/bin/activate"
