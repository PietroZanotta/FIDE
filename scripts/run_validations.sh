#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/run_example_a.sh" --validate-only
"$ROOT/scripts/run_mgd_validation.sh"
"$ROOT/scripts/run_tesseracts.sh"
"$ROOT/scripts/run_ablations.sh"
