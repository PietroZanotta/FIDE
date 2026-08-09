#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"; banner "Independent MGD validation"; run_py validate_mgd.py
