#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${1:-configs/calibration_smoke.yaml}"
OUTPUT="${2:-data/calibration_smoke.npz}"
STEM="$(basename "${OUTPUT}" .npz)"

mbc-generate-data --config "${CONFIG}" --output "${OUTPUT}"
mbc-validate-data "${OUTPUT}" --output "artifacts/${STEM}_validation.json"
mbc-match-regimes "${OUTPUT}" --output "artifacts/${STEM}_matches.json"

echo "validated dataset: ${OUTPUT}"
echo "validation report: artifacts/${STEM}_validation.json"
echo "matched pairs: artifacts/${STEM}_matches.json"
