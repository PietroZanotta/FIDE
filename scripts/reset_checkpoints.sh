#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
mkdir -p "$ROOT/results/example_b"
cp "$ROOT/checkpoints/example_a.npz" "$ROOT/results/learned_mfsi_example_a.npz"
cp "$ROOT/checkpoints/example_b.npz" "$ROOT/results/example_b/learned_mfsi_example_b.npz"
echo "Restored packaged checkpoints."
