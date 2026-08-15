set -euo pipefail

# Run the toy-example and vortex/double-gyre experiment scripts from one launcher.
#
# Usage (from anywhere inside the repository):
#   ./run_toy_and_vortices.sh smoke
#   ./run_toy_and_vortices.sh full
#   ./run_toy_and_vortices.sh audits
#   ./run_toy_and_vortices.sh all
#
# "all" runs, in this exact order:
#   1. vortices full experiment + eval
#   2. toy full experiment + eval
#   3. toy audits
#   4. vortices audits
#
# All stdout/stderr is shown in the terminal and also written to:
#   <repo-root>/full_run.log
#
# Optional environment variables:
#   PYTHON=/path/to/python
#   JAX_ENABLE_X64=1
#   XLA_PYTHON_CLIENT_PREALLOCATE=false
#
# The script assumes the repository contains:
#   src/
#   experiments/toy_example/
#   experiments/vortices/

MODE="${1:-smoke}"
PYTHON="${PYTHON:-python}"

# Resolve repository root robustly.
# If this file is copied to the repo root, SCRIPT_DIR is already the root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/src" && -d "${SCRIPT_DIR}/experiments" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
elif git -C "${SCRIPT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
else
  echo "ERROR: could not locate repository root containing src/ and experiments/." >&2
  echo "Copy this script to the repository root or run it from inside a git checkout." >&2
  exit 2
fi

TOY_DIR="${REPO_ROOT}/experiments/toy_example"
VORTEX_DIR="${REPO_ROOT}/experiments/vortices"
LOG_FILE="${REPO_ROOT}/full_run.log"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Start a fresh combined log for this launcher invocation.
# Everything after this point goes both to terminal and full_run.log.
: > "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required file not found: $f" >&2
    exit 2
  fi
}

run_cmd() {
  echo
  echo "======================================================================"
  echo "+ $*"
  echo "======================================================================"
  "$@"
}

check_layout() {
  require_file "${TOY_DIR}/run.py"
  require_file "${TOY_DIR}/run_gradient_smoke.py"
  require_file "${VORTEX_DIR}/run.py"
  require_file "${VORTEX_DIR}/run_gradient_smoke.py"

  if [[ ! -f "${VORTEX_DIR}/verify_core.py" ]]; then
    echo "WARNING: ${VORTEX_DIR}/verify_core.py not found; core verification will be skipped." >&2
  fi
}

run_smoke() {
  echo "### Core / compatibility checks"
  if [[ -f "${VORTEX_DIR}/verify_core.py" ]]; then
    run_cmd "${PYTHON}" "${VORTEX_DIR}/verify_core.py"
  fi

  echo "### Toy example: exact scientific smoke"
  run_cmd "${PYTHON}" "${TOY_DIR}/run.py" --smoke

  echo "### Toy example: differentiable gradient smoke"
  run_cmd "${PYTHON}" "${TOY_DIR}/run_gradient_smoke.py"

  echo "### Vortices: exact scientific smoke"
  run_cmd "${PYTHON}" "${VORTEX_DIR}/run.py" --smoke

  echo "### Vortices: differentiable gradient smoke"
  run_cmd "${PYTHON}" "${VORTEX_DIR}/run_gradient_smoke.py"
}

run_full_vortices() {
  echo "### Vortices: full experiment"
  run_cmd "${PYTHON}" "${VORTEX_DIR}/run.py"

  if [[ -f "${VORTEX_DIR}/eval.py" ]]; then
    echo "### Vortices: evaluate saved result"
    run_cmd "${PYTHON}" "${VORTEX_DIR}/eval.py"
  else
    echo "SKIP: ${VORTEX_DIR}/eval.py"
  fi
}

run_full_toy() {
  echo "### Toy example: full experiment"
  run_cmd "${PYTHON}" "${TOY_DIR}/run.py"

  if [[ -f "${TOY_DIR}/eval.py" ]]; then
    echo "### Toy example: evaluate saved result"
    run_cmd "${PYTHON}" "${TOY_DIR}/eval.py"
  else
    echo "SKIP: ${TOY_DIR}/eval.py"
  fi
}

run_audits_toy() {
  echo "### Toy example: optional numerical/scientific audits"

  if [[ -f "${TOY_DIR}/run_pareto.py" ]]; then
    run_cmd "${PYTHON}" "${TOY_DIR}/run_pareto.py"
  else
    echo "SKIP: ${TOY_DIR}/run_pareto.py"
  fi

  if [[ -f "${TOY_DIR}/run_proxy_convergence.py" ]]; then
    run_cmd "${PYTHON}" "${TOY_DIR}/run_proxy_convergence.py"
  else
    echo "SKIP: ${TOY_DIR}/run_proxy_convergence.py"
  fi
}

run_audits_vortices() {
  echo "### Vortices: optional numerical/scientific audits"

  if [[ -f "${VORTEX_DIR}/run_pareto.py" ]]; then
    run_cmd "${PYTHON}" "${VORTEX_DIR}/run_pareto.py"
  else
    echo "SKIP: ${VORTEX_DIR}/run_pareto.py"
  fi

  if [[ -f "${VORTEX_DIR}/run_proxy_convergence.py" ]]; then
    run_cmd "${PYTHON}" "${VORTEX_DIR}/run_proxy_convergence.py"
  else
    echo "SKIP: ${VORTEX_DIR}/run_proxy_convergence.py"
  fi
}

run_full() {
  # Requested full-run ordering: vortices first, then toy.
  run_full_vortices
  run_full_toy
}

run_audits() {
  # Requested audit ordering: toy first, then vortices.
  run_audits_toy
  run_audits_vortices
}

main() {
  check_layout

  echo "Repository: ${REPO_ROOT}"
  echo "Python:     ${PYTHON}"
  echo "Mode:       ${MODE}"
  echo "Log:        ${LOG_FILE}"
  echo "Started:    $(date -Is)"

  case "${MODE}" in
    smoke)
      run_smoke
      ;;

    full)
      # 1. vortices full
      # 2. toy full
      run_full
      ;;

    audits)
      # 1. toy audits
      # 2. vortices audits
      run_audits
      ;;

    all)
      # Exact requested order:
      # 1. vortices full
      # 2. toy full
      # 3. toy audits
      # 4. vortices audits
      run_full_vortices
      run_full_toy
      run_audits_toy
      run_audits_vortices
      ;;

    *)
      echo "ERROR: unknown mode '${MODE}'." >&2
      echo "Usage: $0 {smoke|full|audits|all}" >&2
      exit 2
      ;;
  esac

  echo
  echo "All requested ${MODE} commands completed successfully."
  echo "Finished:   $(date -Is)"
  echo "Full log:   ${LOG_FILE}"
}

main "$@"
