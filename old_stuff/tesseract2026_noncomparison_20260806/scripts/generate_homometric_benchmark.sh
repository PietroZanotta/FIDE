#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python -m manybody_completion.cli_generate_homometric \
  --config configs/homometric_benchmark.yaml \
  --output data/homometric_benchmark.npz \
  --report artifacts/homometric_validation.json
