#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="artifacts/backend_pipeline_parity.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --seed) shift 2 ;;
    --transport) shift 2 ;;
    --no-start-services) shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
PYTHONPATH=src python scripts/run_tesseract_backend_smoke.py --output "$OUTPUT"
