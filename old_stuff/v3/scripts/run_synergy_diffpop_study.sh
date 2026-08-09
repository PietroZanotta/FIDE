#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Three independently trained seeds on the deliberately difficult finite-budget
# configuration.  The workflow trains all baselines, the ordinary Full-E2E
# route, and the co-adaptive Synergy-E2E route, then aggregates paired effects.
run_module manybody_completion.workflow_cli seed-sweep \
  --config configs/diffpop_synergy_seed_sweep.yaml \
  "$@"
