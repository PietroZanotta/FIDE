#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements/host-cpu.txt
python -m pip install -e .

mbc-check-env
pytest
printf '\nCPU environment ready. Run: source .venv/bin/activate\n'
