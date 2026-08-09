#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

banner "MFSI Stage 3B: reconstruct ten frozen base models"
run_py "$ROOT/prepare_stage3b_base_models.py" "$@"
