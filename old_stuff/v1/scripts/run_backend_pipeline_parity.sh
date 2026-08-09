#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SEED=20260805
CONFIG="configs/scientific_comparison_micro.yaml"
OUTPUT="artifacts/backend_pipeline_parity"
TESSERACT_TRANSPORT="local_api"
START_LOCAL_SERVICES=true
MANAGED_CONTAINERS=()

cleanup_managed_services() {
  if [[ ${#MANAGED_CONTAINERS[@]} -gt 0 ]]; then
    tesseract teardown "${MANAGED_CONTAINERS[@]}" >/dev/null 2>&1 || true
  fi
}

trap cleanup_managed_services EXIT

endpoint_ready() {
  python -c \
    'import sys, urllib.request; urllib.request.urlopen(sys.argv[1].rstrip("/") + "/openapi.json", timeout=2).read(1)' \
    "$1" >/dev/null 2>&1
}

start_local_service() {
  local image_name="$1"
  local port="$2"
  local service_name="$3"
  local serve_output

  echo "Starting $service_name Tesseract on port $port..."
  if ! serve_output="$(tesseract serve --port "$port" "$image_name")"; then
    echo "Could not start $service_name Tesseract." >&2
    echo "Ensure Docker is running and build the images with:" >&2
    echo "  ./scripts/build_scientific_tesseracts.sh" >&2
    exit 1
  fi
  if [[ ! "$serve_output" =~ \"container_name\"[[:space:]]*:[[:space:]]*\"([^\"]+)\" ]]; then
    echo "Could not parse the container name returned by tesseract serve:" >&2
    echo "$serve_output" >&2
    exit 1
  fi
  MANAGED_CONTAINERS+=("${BASH_REMATCH[1]}")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --transport)
      TESSERACT_TRANSPORT="$2"
      shift 2
      ;;
    --no-start-services)
      START_LOCAL_SERVICES=false
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--seed N] [--config PATH] [--output DIRECTORY] [--transport local_api|url] [--no-start-services]" >&2
      exit 2
      ;;
  esac
done

if [[ "$TESSERACT_TRANSPORT" != "local_api" && "$TESSERACT_TRANSPORT" != "url" ]]; then
  echo "--transport must be local_api or url" >&2
  exit 2
fi

LOCAL_OUTPUT="$OUTPUT/local_jax"
TESSERACT_OUTPUT="$OUTPUT/tesseract_${TESSERACT_TRANSPORT}"

if [[ "$TESSERACT_TRANSPORT" == "url" ]]; then
  RELAXATION_URL="${MBC_RELAXATION_TESSERACT_URL:-http://127.0.0.1:8001}"
  PROJECTION_URL="${MBC_PROJECTION_TESSERACT_URL:-http://127.0.0.1:8002}"
  export MBC_RELAXATION_TESSERACT_URL="$RELAXATION_URL"
  export MBC_PROJECTION_TESSERACT_URL="$PROJECTION_URL"

  if ! endpoint_ready "$RELAXATION_URL"; then
    if [[ "$START_LOCAL_SERVICES" == true && "$RELAXATION_URL" == "http://127.0.0.1:8001" ]]; then
      start_local_service manybody-scientific-relaxation 8001 relaxation
    else
      echo "Relaxation Tesseract is unreachable: $RELAXATION_URL/openapi.json" >&2
      exit 1
    fi
  fi
  if ! endpoint_ready "$PROJECTION_URL"; then
    if [[ "$START_LOCAL_SERVICES" == true && "$PROJECTION_URL" == "http://127.0.0.1:8002" ]]; then
      start_local_service manybody-scientific-projection 8002 projection
    else
      echo "Projection Tesseract is unreachable: $PROJECTION_URL/openapi.json" >&2
      exit 1
    fi
  fi
  if ! endpoint_ready "$RELAXATION_URL" || ! endpoint_ready "$PROJECTION_URL"; then
    echo "Tesseract services did not become ready; aborting before training." >&2
    exit 1
  fi
  echo "Tesseract URL endpoints are ready."
fi

python scripts/run_tesseract_backend_smoke.py \
  --output "$OUTPUT/solver_operator_parity.json"

python -m manybody_completion.comparison_cli \
  --config "$CONFIG" \
  --output "$LOCAL_OUTPUT" \
  --seed "$SEED" \
  --rerun-flow

MBC_SOLVER_BACKEND=tesseract \
MBC_TESSERACT_TRANSPORT="$TESSERACT_TRANSPORT" \
python -m manybody_completion.comparison_cli \
  --config "$CONFIG" \
  --output "$TESSERACT_OUTPUT" \
  --seed "$SEED" \
  --rerun-flow

python scripts/validate_scientific_outputs.py --directory "$LOCAL_OUTPUT"
python scripts/validate_scientific_outputs.py --directory "$TESSERACT_OUTPUT"
python scripts/check_comparison_reproducibility.py \
  --left "$LOCAL_OUTPUT" \
  --right "$TESSERACT_OUTPUT" \
  --output "$OUTPUT/pipeline_parity.json" \
  --allow-backend-difference

echo "Backend pipeline parity passed: $OUTPUT/pipeline_parity.json"
