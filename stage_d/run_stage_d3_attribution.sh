#!/usr/bin/env bash
set -euo pipefail

# Stage D.3 attribution sweep
# ----------------------------
# Decomposes:
#   1) action Monte Carlo variance at the reference finite-resource condition,
#   2) finite-population sampling noise only,
#   3) detector-noise floor (sampling noise made negligible),
#   4) acquisition-time density K.
#
# All runs use the same D0 checkpoint, same D2 FM particle construction,
# same seed, and 24 common-random-number scientific trials.

PYTHON="${PYTHON:-python}"

D3="${D3:-./stage_d3_flow_matching_finite_measurements.py}"
BACKEND="${BACKEND:-../stage_b/stage_b2_transport_conditioned_design.py}"
C2="${C2:-../stage_c/stage_c2_mfsi_matched_action.py}"
D0="${D0:-./stage_d0_flow_matching_reference.py}"
D2="${D2:-./stage_d2_flow_matching_particle_mfsi.py}"
CHECKPOINT="${CHECKPOINT:-./stage_d0_flow_matching_reference.npz}"
OUTDIR="${OUTDIR:-./d3_attribution_runs}"
SEED="${SEED:-20260812}"

mkdir -p "$OUTDIR"

for f in "$D3" "$BACKEND" "$C2" "$D0" "$D2" "$CHECKPOINT"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required file not found: $f" >&2
    echo "Override its path with the corresponding environment variable." >&2
    exit 2
  fi
done

COMMON=(
  "$D3"
  --backend "$BACKEND"
  --c2-script "$C2"
  --d0-script "$D0"
  --d2-script "$D2"
  --checkpoint "$CHECKPOINT"
  --preset reference
  --seed "$SEED"
  --trials 24
  --action-trials 24
)

run_case () {
  local label="$1"; shift
  local outfile="$OUTDIR/${label}.json"
  local logfile="$OUTDIR/${label}.log"
  echo
  echo "================================================================================"
  echo "RUN: $label"
  echo "OUTPUT: $outfile"
  echo "================================================================================"
  "$PYTHON" "${COMMON[@]}" "$@" --output "$outfile" 2>&1 | tee "$logfile"
}

# -----------------------------------------------------------------------------
# 1. Reference finite-resource condition, now with all 24 full-action trials.
#    This is the primary run for diagnosing the large Lift action variance.
# -----------------------------------------------------------------------------
run_case stage_d3_action24 \
  --N 100 \
  --K 11 \
  --noise-std 0.01

# K=11 is exactly the reference run above. Keep a consistently named copy for
# the K-sweep summary without spending another full run.
cp -f "$OUTDIR/stage_d3_action24.json" "$OUTDIR/stage_d3_K11.json"
cp -f "$OUTDIR/stage_d3_action24.log"  "$OUTDIR/stage_d3_K11.log"

# -----------------------------------------------------------------------------
# 2. Sampling noise only: finite N, zero detector noise.
# -----------------------------------------------------------------------------
run_case stage_d3_sampling_only \
  --N 100 \
  --K 11 \
  --noise-std 0.0

# -----------------------------------------------------------------------------
# 3. Detector-noise floor: N is made very large so Sigma/N is negligible
#    relative to detector variance 0.01^2. Exact endpoints remain anchored.
# -----------------------------------------------------------------------------
run_case stage_d3_detector_floor \
  --N 100000 \
  --K 11 \
  --noise-std 0.01

# -----------------------------------------------------------------------------
# 4. Temporal acquisition-density sweep at fixed N=100 and detector noise=0.01.
#    K=11 is already stage_d3_action24 above.
# -----------------------------------------------------------------------------
for K in 5 7 17; do
  run_case "stage_d3_K${K}" \
    --N 100 \
    --K "$K" \
    --noise-std 0.01
done

# -----------------------------------------------------------------------------
# Compact cross-run attribution summary.
# -----------------------------------------------------------------------------
"$PYTHON" - "$OUTDIR" <<'PY'
import json
import math
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
files = [
    ("reference", outdir / "stage_d3_action24.json"),
    ("sampling_only", outdir / "stage_d3_sampling_only.json"),
    ("detector_floor", outdir / "stage_d3_detector_floor.json"),
    ("K5", outdir / "stage_d3_K5.json"),
    ("K7", outdir / "stage_d3_K7.json"),
    ("K11", outdir / "stage_d3_K11.json"),
    ("K17", outdir / "stage_d3_K17.json"),
]

def mean(p, design, key):
    return float(p["design_summary"][design][key]["mean"])

def se(p, design, key):
    return float(p["design_summary"][design][key]["se"])

rows = []
for label, path in files:
    p = json.loads(path.read_text())
    lift_L = mean(p, "lift", "learned_finite_heldout_mmd2")
    full_L = mean(p, "full", "learned_finite_heldout_mmd2")
    lift_dL = mean(p, "lift", "learned_measurement_delta_mmd2")
    full_dL = mean(p, "full", "learned_measurement_delta_mmd2")
    lift_A = mean(p, "lift", "learned_finite_action")
    full_A = mean(p, "full", "learned_finite_action")
    c = p["contrasts"]["learned_full_vs_lift_measurement_degradation"]
    a = p["contrasts"]["learned_full_vs_lift_finite_action_reduction"]
    proj = p["projection_summary"]
    attr = p.get("attribution_diagnostics", {})
    rows.append({
        "label": label,
        "N": p["condition"]["N"],
        "K": p["condition"]["K"],
        "noise": p["condition"]["obs_noise_std"],
        "lift_L": lift_L,
        "full_L": full_L,
        "full_law_penalty_pct": 100.0 * (full_L / lift_L - 1.0),
        "lift_dL": lift_dL,
        "full_dL": full_dL,
        "diff_dL": float(c["mean_difference_a_minus_b"]),
        "diff_dL_se": float(c["se_difference"]),
        "lift_A": lift_A,
        "lift_A_se": se(p, "lift", "learned_finite_action"),
        "full_A": full_A,
        "full_A_se": se(p, "full", "learned_finite_action"),
        "action_reduction_pct": 100.0 * float(a["ratio_of_means_reduction"]),
        "paired_action_reduction_pct": 100.0 * float(a["mean_paired_reduction"]),
        "paired_action_reduction_se_pct": 100.0 * float(a["se_paired_reduction"]),
        "projection_pct": 100.0 * float(proj["active_fraction"]),
        "projection_cases": int(attr.get("projection_case_count", 0)),
    })

print("\n" + "=" * 160)
print("D3 ATTRIBUTION SUMMARY")
print("=" * 160)
head = (
    f"{'case':15s} {'N':>7s} {'K':>3s} {'noise':>7s} "
    f"{'Full/Lift L%':>13s} {'Lift dL':>11s} {'Full dL':>11s} {'diff dL':>12s} "
    f"{'Lift A':>12s} {'Full A':>12s} {'A red.%':>9s} {'paired%':>9s} {'proj%':>7s}"
)
print(head)
print("-" * len(head))
for r in rows:
    print(
        f"{r['label']:15s} {r['N']:7d} {r['K']:3d} {r['noise']:7.3g} "
        f"{r['full_law_penalty_pct']:13.3f} {r['lift_dL']:11.3e} {r['full_dL']:11.3e} {r['diff_dL']:12.3e} "
        f"{r['lift_A']:12.2f} {r['full_A']:12.2f} {r['action_reduction_pct']:9.2f} "
        f"{r['paired_action_reduction_pct']:9.2f} {r['projection_pct']:7.2f}"
    )

summary_path = outdir / "stage_d3_attribution_summary.json"
summary_path.write_text(json.dumps(rows, indent=2) + "\n")
print("=" * 160)
print(f"Saved compact summary: {summary_path}")

# Print projected cases for the main reference run.
main = json.loads((outdir / "stage_d3_action24.json").read_text())
cases = main.get("attribution_diagnostics", {}).get("projection_cases", [])
print("\nProjected cases in reference run:")
if not cases:
    print("  none")
else:
    for c in cases:
        print(
            f"  {c['design']:8s} trial={c['trial_index_1based']:2d} alpha={c['alpha']:.5f} "
            f"proj={c['projection_norm']:.3e} cal={c['learned_finite_max_calibration_resid']:.3e} "
            f"ESS={c['learned_finite_min_ess']:.3f} A={c['learned_finite_action']:.2f}"
        )

print("\nTop learned finite-action cases in reference run:")
for design in ("lift", "tangent", "full"):
    top = main.get("attribution_diagnostics", {}).get("top_learned_action_cases", {}).get(design, [])
    print(f"  {design}:")
    for r in top[:5]:
        print(
            f"    trial={r['trial_index_1based']:2d} alpha={r['alpha']:.5f} "
            f"A={r['learned_finite_action']:.2f} infl={100*r['measurement_action_inflation']:+.2f}% "
            f"proj={r['projection_active']} cal={r['calibration_residual']:.2e} ESS={r['min_ess']:.3f}"
        )
PY

echo
echo "All D3 attribution runs completed."
echo "Results directory: $OUTDIR"
