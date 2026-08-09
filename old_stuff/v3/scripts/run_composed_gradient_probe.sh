#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

OUTPUT="artifacts/composed_gradient_probe.json"
CONFIG="configs/diffpop_micro.yaml"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
run_module manybody_completion.workflow_cli gradient-probe \
  --config "$CONFIG" --output "$OUTPUT" "${ARGS[@]}"
