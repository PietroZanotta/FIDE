#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
banner "Tesseract Core systems benchmark"
run_py tesseract_systems_benchmark.py "$@"
