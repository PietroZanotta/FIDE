#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUICK=0; RETRAIN=0; BACKEND="${MFSI_BACKEND:-tesseract}"
usage(){ echo "Usage: ./scripts/run_all.sh [--backend tesseract|jax] [--quick] [--retrain]"; }
while [[ $# -gt 0 ]]; do case "$1" in
 --backend) BACKEND="$2"; shift 2;; --quick) QUICK=1; shift;; --retrain) RETRAIN=1; shift;;
 -h|--help) usage; exit 0;; *) usage >&2; exit 2;; esac; done
A=(--backend "$BACKEND"); B=(--backend "$BACKEND")
[[ "$RETRAIN" -eq 0 ]] || { A+=(--retrain); B+=(--retrain); }
[[ "$QUICK" -eq 0 ]] || B+=(--quick)
"$ROOT/scripts/run_example_a.sh" "${A[@]}"
"$ROOT/scripts/run_mgd_validation.sh"
"$ROOT/scripts/run_ablations.sh"
"$ROOT/scripts/run_tesseracts.sh"
"$ROOT/scripts/run_example_b.sh" "${B[@]}"
