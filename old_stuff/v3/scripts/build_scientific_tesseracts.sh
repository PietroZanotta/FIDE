#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for component in scientific_tilted_ensemble scientific_dual_calibration; do
  directory="tesseracts/${component}"
  test -f "${directory}/tesseract_api.py"
  test -f "${directory}/tesseract_config.yaml"
  test -f "${directory}/tesseract_requirements.txt"
  echo "validated ${directory}"
done
echo "Packaged Tesseract component files are present."
