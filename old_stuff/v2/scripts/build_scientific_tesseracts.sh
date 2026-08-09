#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for component in scientific_tilted_ensemble scientific_dual_calibration; do
  test -f "tesseracts/${component}/tesseract_api.py"
  test -f "tesseracts/${component}/tesseract_config.yaml"
  test -f "tesseracts/${component}/tesseract_requirements.txt"
  echo "validated tesseracts/${component}"
done
echo "Install a Tesseract runtime separately to build/serve these packaged APIs."
