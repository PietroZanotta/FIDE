#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
./scripts/run_tests.sh
./scripts/build_scientific_tesseracts.sh
./scripts/run_from_scratch_acceptance.sh
./scripts/run_backend_pipeline_parity.sh \
  --output artifacts/diffpop_micro/tesseract_backend_smoke.json
