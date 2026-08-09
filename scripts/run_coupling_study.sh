#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ARGS=()
usage(){
  cat <<'EOF'
Usage: ./scripts/run_coupling_study.sh [--quick]
       [--seeds "401 402 403 404 405"] [--no-plots]
       [--aggregate-existing]

Run Stage 2 only: independent, geometric Sinkhorn, and fiber-aware endpoint
couplings with each paper schedule frozen. This command never performs joint
schedule-coupling optimization.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick|--no-plots|--aggregate-existing) ARGS+=("$1"); shift;;
    --seeds) ARGS+=("$1" "$2"); shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

banner "MFSI Stage 2: frozen-schedule coupling study"
run_py "$ROOT/coupling_study.py" "${ARGS[@]}"
