#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"; banner "Differentiation ablations / microbenchmarks"; run_py ablate_and_benchmark.py
