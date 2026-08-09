#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements/host-cpu.txt
python -m pip install -e .

python -c 'import jax, manybody_completion, tesseract_core, tesseract_jax'
pytest tests/test_scientific_numerics.py tests/test_seed_study.py tests/test_uq.py
printf '\nCPU environment ready. Run: source .venv/bin/activate\n'
