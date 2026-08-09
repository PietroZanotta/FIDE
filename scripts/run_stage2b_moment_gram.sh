#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ARGS=()
usage(){
  cat <<'EOF'
Usage: ./scripts/run_stage2b_moment_gram.sh [--aggregate-existing] [--no-plots]

Run exactly one Stage 2B follow-up: retain geometric and Phi interactions and
add the fixed 36-parameter moment-response Gram interaction. Schedules remain
frozen, and this command never performs joint schedule-coupling optimization.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --aggregate-existing|--no-plots) ARGS+=("$1"); shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

banner "MFSI Stage 2B: one moment-response Gram coupling follow-up"
run_py "$ROOT/stage2b_moment_gram_coupling.py" "${ARGS[@]}"
