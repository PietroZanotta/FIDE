#!/usr/bin/env python3
"""Fail fast if required generated deliverables are absent or internally weak."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/ocean_drifters/analysis"
REQUIRED = [
    "README.md", "data_audit.md", "recommendation.md", "gates.yaml",
    "tables/data_audit.csv", "tables/gate_horizon_counts.csv",
    "tables/candidate_metrics.csv", "tables/exclusions_summary.csv",
    "figures/coverage/global_observation_density.png",
    "figures/coverage/wna_unique_drifter_density.png",
    "figures/trajectories/north_atlantic_A_45d_spaghetti.png",
    "figures/snapshots/north_atlantic_A_45d_snapshots.png",
    "figures/dispersion/north_atlantic_A_45d_dispersion.png",
    "figures/seasonality/north_atlantic_A_45d_by_season.png",
    "figures/seasonality/north_atlantic_A_45d_by_decade.png",
]
missing = [name for name in REQUIRED if not (OUT / name).is_file()]
assert not missing, f"missing deliverables: {missing}"
with (OUT / "tables/candidate_metrics.csv").open(newline="", encoding="utf-8") as handle:
    metrics = list(csv.DictReader(handle))
winner = next(row for row in metrics if row["gate"] == "north_atlantic_A" and row["horizon_days"] == "45")
assert int(winner["N_total"]) >= 300
report = (OUT / "recommendation.md").read_text(encoding="utf-8")
for phrase in ["north_atlantic_A", "45-day", "all-season", "Lambert azimuthal equal-area", "Caveats"]:
    assert phrase in report, phrase
print(f"Validated {len(REQUIRED)} required deliverables; selected N={winner['N_total']}.")
