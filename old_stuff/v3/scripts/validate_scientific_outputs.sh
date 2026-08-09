#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
if [[ $# -ne 2 || "$1" != "--directory" ]]; then
  echo "usage: $0 --directory OUTPUT_DIR" >&2
  exit 2
fi
run_module manybody_completion.workflow_cli validate --directory "$2"
