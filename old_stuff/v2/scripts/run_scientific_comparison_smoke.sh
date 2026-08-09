#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src python -m manybody_completion.comparison_cli \
  --config configs/diffpop_smoke.yaml \
  --output artifacts/diffpop_smoke "$@"
