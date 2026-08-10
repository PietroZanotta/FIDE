#!/usr/bin/env python3

import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean

n=6 # seed che vogliamo vedere

RUN = Path("results/observable_design_toy/confirmatory/R3")
CELLS = RUN / "cells"

pattern = re.compile(
    r"nominal_(.+)_model_(\d+)_eval_(\d+)\.json$"
)

records = []

for path in CELLS.glob("nominal_*_model_*_eval_*.json"):
    m = pattern.match(path.name)
    if not m:
        continue

    objective = m.group(1)
    model_seed = int(m.group(2))
    eval_seed = int(m.group(3))

    data = json.loads(path.read_text())
    d = data["downstream"]

    records.append({
        "objective": objective,
        "model_seed": model_seed,
        "eval_seed": eval_seed,

        "tangent_local_mmd":
            d["local_summary"]["mean_tangent_next_mmd"],

        "velocity_gap":
            d["local_summary"]["mean_velocity_gap_mse"],

        "tangent_rollout_mmd":
            d["summary"]["moment_tangent"]["mean_interior_mmd"],

        "mfsi_rollout_mmd":
            d["summary"]["mfsi_learned_safe"]["mean_interior_mmd"],

        "max_moment_error":
            d["summary"]["mfsi_learned_safe"]["max_moment_error"],

        "angular_error":
            d["summary"]["mfsi_learned_safe"]["mean_interior_angular_error"],
    })


if not records:
    raise SystemExit(f"No completed cells found in {CELLS}")


# Lowest model seed = first model seed.
first_seed = sorted({r["model_seed"] for r in records})

if len(first_seed) < n:
    raise SystemExit(
        f"Seed {n} is not available yet. Completed model seeds: {first_seed}"
    )

selected_seed = first_seed[n-1]

records = [
    r for r in records
    if r["model_seed"] == selected_seed
]

print(f"\nSECOND MODEL SEED: {selected_seed}")

print(f"\nFIRST MODEL SEED: {first_seed}")
print("=" * 70)


# Group evaluation banks by objective.
groups = defaultdict(list)
for r in records:
    groups[r["objective"]].append(r)


metrics = [
    "tangent_local_mmd",
    "velocity_gap",
    "tangent_rollout_mmd",
    "mfsi_rollout_mmd",
    "max_moment_error",
    "angular_error",
]


for objective, rows in sorted(groups.items()):
    print(f"\n{objective.upper()}")
    print(f"  completed evaluation banks: {len(rows)}")
    print(f"  eval seeds: {[r['eval_seed'] for r in rows]}")

    for metric in metrics:
        vals = [float(r[metric]) for r in rows]

        print(
            f"  {metric:24s} "
            f"mean={mean(vals):10.5g}  "
            f"min={min(vals):10.5g}  "
            f"max={max(vals):10.5g}"
        )


print("\n" + "=" * 70)
print("RANKINGS (mean over completed evaluation banks)")
print("lower is better")
print("=" * 70)

for metric in metrics:
    ranking = []

    for objective, rows in groups.items():
        value = mean(float(r[metric]) for r in rows)
        ranking.append((value, objective))

    ranking.sort()

    print(f"\n{metric}")
    for i, (value, objective) in enumerate(ranking, 1):
        print(f"  {i}. {objective:12s} {value:.6g}")