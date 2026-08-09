#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m manybody_completion.comparison_cli \
  --config configs/scientific_comparison_smoke.yaml \
  --output artifacts/scientific_comparison_smoke "$@"
