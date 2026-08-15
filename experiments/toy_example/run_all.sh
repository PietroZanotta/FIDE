#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  echo "usage: $0 --smoke | --run" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage

case "$1" in
  --smoke)
    exec python -u "${HERE}/run.py" --smoke
    ;;
  --run)
    # Proper run uses configuration only; the Python CLI gets no scientific flags.
    exec python -u "${HERE}/run.py"
    ;;
  *)
    usage
    ;;
esac
