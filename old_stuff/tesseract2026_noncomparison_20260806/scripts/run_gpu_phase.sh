#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_gpu_phase.sh pretrain -- <command> [args...]
  run_gpu_phase.sh e2e      -- <command> [args...]

pretrain:
  Gives the native JAX process most GPU memory. Do not run GPU Tesseract
  containers concurrently.

e2e:
  Leaves memory for persistent Tesseract services. For the N=4 homometric
  benchmark, CPU Tesseracts are often faster overall than sharing the GPU.
EOF
}

[[ $# -ge 3 ]] || { usage; exit 2; }
PHASE="$1"
shift
[[ "$1" == "--" ]] || { usage; exit 2; }
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$ROOT/.jax_cache}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-1}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-tensorfloat32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

case "$PHASE" in
  pretrain)
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.88}"
    ;;
  e2e)
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
    ;;
  *)
    usage
    exit 2
    ;;
esac

exec "$@"
