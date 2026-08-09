#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG=""
OUTPUT=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ -z "$CONFIG" || -z "$OUTPUT" ]]; then
  echo "usage: $0 --config CONFIG.yaml --output OUTPUT_DIR [comparison options]" >&2
  exit 2
fi
run_module manybody_completion.comparison_cli \
  --config "$CONFIG" --output "$OUTPUT" "${ARGS[@]}"
run_module manybody_completion.workflow_cli validate --directory "$OUTPUT"
