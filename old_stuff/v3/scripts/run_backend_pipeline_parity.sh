#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

OUTPUT="artifacts/tesseract_backend_smoke.json"
CONFIG="configs/diffpop_micro.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
run_module manybody_completion.workflow_cli backend-smoke \
  --config "$CONFIG" --output "$OUTPUT"
