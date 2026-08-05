#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f data/homometric_benchmark.npz ]]; then
  ./scripts/generate_homometric_benchmark.sh
fi

python experiments/flow_matching/run_homometric_flow.py
