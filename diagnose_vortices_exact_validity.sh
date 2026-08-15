#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
V="${ROOT}/experiments/vortices"
OUT="${V}/outputs/run"

echo "=== Vortex exact-validity diagnostic ==="
echo "repo: ${ROOT}"
echo

echo "=== Relevant config blocks ==="
python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/vortices/config.json")
c = json.loads(p.read_text())
for key in ("measurement", "projection", "law", "optimization", "reference", "poisson"):
    if key in c:
        print(f"\n[{key}]")
        print(json.dumps(c[key], indent=2))
PY

echo
echo "=== Reference-bank support inside [0,2] x [0,1] ==="
python - <<'PY'
from pathlib import Path
import numpy as np

p = Path("experiments/vortices/outputs/run/reference_bank.npz")
if not p.exists():
    print(f"missing: {p}")
    raise SystemExit(0)

z = np.load(p, allow_pickle=False)
print("keys:", sorted(z.files))

node_key = next((k for k in ("nodes", "reference_nodes", "x", "particles") if k in z.files), None)
weight_key = next((k for k in ("base_weights", "weights", "reference_weights") if k in z.files), None)

if node_key is None:
    print("Could not identify reference particle array automatically.")
    raise SystemExit(0)

x = np.asarray(z[node_key], dtype=float)
print("nodes shape:", x.shape)

inside = (
    (x[..., 0] >= 0.0) & (x[..., 0] <= 2.0) &
    (x[..., 1] >= 0.0) & (x[..., 1] <= 1.0)
)

if weight_key is None:
    mass = inside.mean(axis=-1)
    print("No weight array identified; unweighted in-domain fractions:")
else:
    w = np.asarray(z[weight_key], dtype=float)
    print("weights shape:", w.shape)
    if w.ndim == 1:
        w = w / w.sum()
        mass = np.sum(inside * w, axis=-1)
    elif w.ndim == 2 and w.shape == inside.shape:
        w = w / np.maximum(w.sum(axis=-1, keepdims=True), 1e-300)
        mass = np.sum(inside * w, axis=-1)
    else:
        print("Unexpected weight shape; reporting unweighted fractions instead.")
        mass = inside.mean(axis=-1)

mass = np.asarray(mass)
print("in-domain mass/fraction by time:")
print(mass)
print("minimum:", float(np.min(mass)))
print("maximum:", float(np.max(mass)))
PY

echo
echo "=== Selection audit implementation ==="
grep -n -A120 -B10 "def _audit_population" "${V}/selection.py" || true

echo
echo "=== Exact population evaluator / validity gates ==="
grep -n -E -A80 -B10 \
  "def exact_population|def .*population.*result|max_.*calibration|min_ess|ess_fraction|valid" \
  "${V}/experiment.py" | head -n 260 || true

echo
echo "=== Existing selection/cache artifacts ==="
find "${OUT}" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort || true
