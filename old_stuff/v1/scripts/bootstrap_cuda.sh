#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements/host-cuda.txt
python -m pip install -e .

python -c 'import jax, manybody_completion, tesseract_core, tesseract_jax; print(jax.devices())'
printf '\nEnvironment ready. Run: source .venv/bin/activate\n'
