#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
./scripts/run_from_scratch_acceptance.sh
PYTHONPATH=src python scripts/run_tesseract_backend_smoke.py \
  --output artifacts/diffpop_micro/tesseract_backend_smoke.json
