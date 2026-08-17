#!/usr/bin/env python3
"""Rebuild, project, audit, and split the frozen 45-day NOAA cohort."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import load_data  # noqa: E402
from phase2_common import (  # noqa: E402
    load_phase2_config,
    resolve,
    seasons_from_month,
    sha256,
    write_csv,
    write_json,
)


def first_west_entry(lon: np.ndarray, lat: np.ndarray, bounds: list[float]) -> int | None:
    xmin, xmax, ymin, ymax = bounds
    inside = (
        np.isfinite(lon) & np.isfinite(lat)
        & (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
    )
    entries = np.flatnonzero(inside & np.r_[False, ~inside[:-1]])
    entries = entries[(entries > 0) & (lon[entries - 1] < xmin)]
    return int(entries[0]) if entries.size else None


def stratified_split(season: np.ndarray, decade: np.ndarray, sizes: dict[str, int], seed: int) -> np.ndarray:
    names = list(sizes)
    capacities = np.array([sizes[name] for name in names], dtype=int)
    proportions = capacities / capacities.sum()
    assignment = np.empty(len(season), dtype="U12")
    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, key in enumerate(zip(season.tolist(), decade.tolist(), strict=True)):
        strata[key].append(i)
    rng = np.random.default_rng(seed)
    remaining = capacities.copy()
    for key in sorted(strata):
        indices = np.asarray(strata[key], dtype=int)
        rng.shuffle(indices)
        local = np.zeros(len(names), dtype=int)
        for position, index in enumerate(indices, start=1):
            desired = proportions * position
            score = desired - local
            score[remaining <= 0] = -np.inf
            chosen = int(np.argmax(score))
            assignment[index] = names[chosen]
            local[chosen] += 1
            remaining[chosen] -= 1
    # Correct rare global rounding drift by moving within strata whenever possible.
    while np.any(remaining != 0):
        receiver = int(np.flatnonzero(remaining > 0)[0])
        donor = int(np.flatnonzero(remaining < 0)[0])
        donor_indices = np.flatnonzero(assignment == names[donor])
        index = int(donor_indices[-1])
        assignment[index] = names[receiver]
        remaining[receiver] -= 1
        remaining[donor] += 1
    assert Counter(assignment) == Counter(sizes)
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_phase2_config(args.config)
    cohort_cfg = cfg["cohort"]
    processed = resolve(cfg["processed_dir"])
    analysis = resolve(cfg["analysis_dir"])
    figures = analysis / "figures"
    tables = analysis / "tables"
    for directory in [processed / "splits", figures / "cohort", figures / "splits", tables]:
        directory.mkdir(parents=True, exist_ok=True)

    raw_path = resolve(cfg["raw_input"])
    print(f"[cohort] reading canonical raw data: {raw_path}", flush=True)
    raw = load_data(raw_path)
    expected_points = int(cohort_cfg["observations"])
    spacing = int(cohort_cfg["spacing_seconds"])
    horizon_seconds = int(cohort_cfg["horizon_days"] * 86400)
    records = []
    lonlat = []
    absolute_time = []
    for i, drifter_id in enumerate(raw.ids):
        a, b = int(raw.offsets[i]), int(raw.offsets[i + 1])
        k = first_west_entry(raw.lon[a:b], raw.lat[a:b], cohort_cfg["gate_bounds_lon_lat"])
        if k is None:
            continue
        start = a + k
        end = start + expected_points
        if end > b:
            continue
        times = raw.time[start:end]
        if times[-1] - times[0] != horizon_seconds or not np.all(np.diff(times) == spacing):
            continue
        coordinates = np.column_stack([raw.lon[start:end], raw.lat[start:end]])
        if not np.all(np.isfinite(coordinates)) or not np.all(raw.drogued[start:end]):
            continue
        timestamp = np.datetime64(int(times[0]), "s")
        year = int(str(timestamp)[:4]); month = int(str(timestamp)[5:7])
        records.append({
            "drifter_id": int(drifter_id), "crossing_time": str(timestamp),
            "year": year, "month": month, "decade": year // 10 * 10,
            "raw_trajectory_row": i, "raw_start_offset": start,
            "all_finite": True, "all_drogued": True, "exact_6_hour": True,
        })
        lonlat.append(coordinates)
        absolute_time.append(times.astype(np.int64))

    if len(records) != int(cohort_cfg["expected_n"]):
        raise RuntimeError(
            f"Frozen cohort mismatch: reconstructed {len(records)}, expected {cohort_cfg['expected_n']}. "
            "Downstream pipeline stopped."
        )
    ids = np.asarray([row["drifter_id"] for row in records], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise AssertionError("one-segment-per-drifter invariant failed")
    lonlat = np.asarray(lonlat, dtype=np.float64)
    absolute_time = np.asarray(absolute_time, dtype=np.int64)
    transformer = Transformer.from_crs("EPSG:4326", cohort_cfg["projection"], always_xy=True)
    x, y = transformer.transform(lonlat[..., 0], lonlat[..., 1])
    X = np.stack([x, y], axis=-1) / 1000.0
    relative_days = np.arange(expected_points, dtype=np.float64) * spacing / 86400.0
    normalized_time = relative_days / float(cohort_cfg["horizon_days"])
    year = np.asarray([row["year"] for row in records], dtype=np.int16)
    month = np.asarray([row["month"] for row in records], dtype=np.int8)
    decade = (year // 10 * 10).astype(np.int16)
    season = seasons_from_month(month)
    np.savez_compressed(
        processed / "cohort_45d.npz", X=X, ids=ids, crossing_time=absolute_time[:, 0],
        absolute_time=absolute_time, relative_days=relative_days,
        normalized_time=normalized_time, year=year, month=month, decade=decade,
        season=season, projection=np.asarray(cohort_cfg["projection"]),
    )
    for row, label in zip(records, season, strict=True):
        row["season"] = str(label)
    write_csv(processed / "metadata.csv", records)

    exploratory = np.asarray(cfg["domain"]["exploratory_box_km"], dtype=float)
    final = np.asarray(cfg["domain"]["final_box_km"], dtype=float)
    def outside(box):
        return ((X[..., 0] < box[0]) | (X[..., 0] > box[1]) | (X[..., 1] < box[2]) | (X[..., 1] > box[3]))
    outside_old = outside(exploratory); outside_final = outside(final)
    flat = X.reshape(-1, 2)
    quantile_rows = []
    for probability in [.95, .99, .995, 1.0]:
        tail = (1 - probability) / 2
        bounds = np.quantile(flat, [tail, 1 - tail], axis=0)
        mins, maxs = X.min(axis=1), X.max(axis=1)
        trajectory_bounds = np.array([
            np.quantile(mins, tail, axis=0), np.quantile(maxs, 1 - tail, axis=0)
        ])
        complete = ((mins >= trajectory_bounds[0]) & (maxs <= trajectory_bounds[1])).all(axis=1)
        quantile_rows.append({
            "probability": probability,
            "observation_xmin_km": bounds[0, 0], "observation_xmax_km": bounds[1, 0],
            "observation_ymin_km": bounds[0, 1], "observation_ymax_km": bounds[1, 1],
            "trajectory_xmin_km": trajectory_bounds[0, 0], "trajectory_xmax_km": trajectory_bounds[1, 0],
            "trajectory_ymin_km": trajectory_bounds[0, 1], "trajectory_ymax_km": trajectory_bounds[1, 1],
            "complete_trajectories_contained": int(complete.sum()),
        })
    write_csv(tables / "domain_diagnostics.csv", quantile_rows)
    extent_excess = np.maximum.reduce([
        exploratory[0] - X[..., 0], X[..., 0] - exploratory[1],
        exploratory[2] - X[..., 1], X[..., 1] - exploratory[3],
    ])
    extreme_order = np.argsort(extent_excess.max(axis=1))[::-1]
    extreme_rows = []
    for i in extreme_order[:15]:
        if not outside_old[i].any():
            continue
        extreme_rows.append({
            "drifter_id": int(ids[i]), "outside_observations": int(outside_old[i].sum()),
            "xmin_km": X[i, :, 0].min(), "xmax_km": X[i, :, 0].max(),
            "ymin_km": X[i, :, 1].min(), "ymax_km": X[i, :, 1].max(),
        })
    write_csv(tables / "domain_extreme_trajectories.csv", extreme_rows)
    domain_report = f"""# Phase 2 computational-domain decision

The frozen cohort reproduces exactly: **N={len(X)}**, 181 finite, drogued six-hour observations per trajectory. No trajectory is removed.

The exploratory box `x=[{exploratory[0]:.0f},{exploratory[1]:.0f}] km`, `y=[{exploratory[2]:.0f},{exploratory[3]:.0f}] km` excludes {outside_old.mean():.4%} of observations and is exited by {outside_old.any(axis=1).sum()}/{len(X)} trajectories ({outside_old.any(axis=1).mean():.2%}). It therefore cannot be used with silent clipping.

The empirical extrema are x=[{flat[:,0].min():.1f},{flat[:,0].max():.1f}] km and y=[{flat[:,1].min():.1f},{flat[:,1].max():.1f}] km. The 99.5% observation box is x=[{quantile_rows[2]['observation_xmin_km']:.1f},{quantile_rows[2]['observation_xmax_km']:.1f}] km and y=[{quantile_rows[2]['observation_ymin_km']:.1f},{quantile_rows[2]['observation_ymax_km']:.1f}] km. Exact quantile and complete-trajectory boxes are in [domain_diagnostics.csv](tables/domain_diagnostics.csv); responsible IDs are in [domain_extreme_trajectories.csv](tables/domain_extreme_trajectories.csv).

## Frozen decision

Use **x=[{final[0]:.0f},{final[1]:.0f}] km, y=[{final[2]:.0f},{final[3]:.0f}] km** ({final[1]-final[0]:.0f}×{final[3]-final[2]:.0f} km). This modest enlargement contains {1-outside_final.mean():.8%} of observations and {len(X)-outside_final.any(axis=1).sum()}/{len(X)} complete trajectories. It includes every observed excursion with at least {min(flat[:,0].min()-final[0], final[1]-flat[:,0].max(), flat[:,1].min()-final[2], final[3]-flat[:,1].max()):.1f} km coordinate margin. No empirical mass is clipped and no extreme drifter is deleted.
"""
    (analysis / "domain_decision.md").write_text(domain_report, encoding="utf-8")

    split_sizes = cfg["splits"]
    sizes = {name: int(split_sizes[name]) for name in ["inference", "validation", "final_test"]}
    assignment = stratified_split(season, decade, sizes, int(cfg["seed"]))
    split_rows = []
    for i, split in enumerate(assignment):
        split_rows.append({
            "drifter_id": int(ids[i]), "split": str(split), "season": str(season[i]),
            "year": int(year[i]), "decade": int(decade[i]),
            "crossing_time": str(np.datetime64(int(absolute_time[i, 0]), "s")),
        })
    write_csv(processed / "splits/split_manifest.csv", split_rows)
    for split in sizes:
        selected = ids[assignment == split]
        (processed / f"splits/{split}_ids.txt").write_text(
            "\n".join(str(int(value)) for value in selected) + "\n", encoding="utf-8"
        )
    dev_mask = assignment != "final_test"
    np.savez_compressed(
        processed / "development_270.npz", X=X[dev_mask], ids=ids[dev_mask],
        split=assignment[dev_mask], season=season[dev_mask], year=year[dev_mask],
        decade=decade[dev_mask], relative_days=relative_days,
        normalized_time=normalized_time, domain_km=final,
    )
    development_indices = np.flatnonzero(dev_mask)
    cv_rows = []
    # Repeat 0 is exactly the permanent primary 200/70 partition.
    for local_index, cohort_index in enumerate(development_indices):
        cv_rows.append({
            "repeat": 0, "drifter_id": int(ids[cohort_index]),
            "role": str(assignment[cohort_index]), "season": str(season[cohort_index]),
            "decade": int(decade[cohort_index]),
        })
    for repeat in range(1, int(split_sizes["repeated_cv_folds"])):
        cv_assignment = stratified_split(
            season[dev_mask], decade[dev_mask], {"inference": 200, "validation": 70},
            int(cfg["seed"]) + int(split_sizes["cv_seed_offset"]) + repeat,
        )
        for local_index, cohort_index in enumerate(development_indices):
            cv_rows.append({
                "repeat": repeat, "drifter_id": int(ids[cohort_index]),
                "role": str(cv_assignment[local_index]), "season": str(season[cohort_index]),
                "decade": int(decade[cohort_index]),
            })
    write_csv(processed / "splits/repeated_cv_manifest.csv", cv_rows)
    for repeat in range(int(split_sizes["repeated_cv_folds"])):
        repeat_rows = [row for row in cv_rows if row["repeat"] == repeat]
        assert Counter(row["role"] for row in repeat_rows) == Counter({"inference": 200, "validation": 70})
        assert len({row["drifter_id"] for row in repeat_rows}) == 270
    split_summary = []
    for split in sizes:
        mask = assignment == split
        for grouping, labels in [("season", season), ("decade", decade.astype(str))]:
            for label, count in sorted(Counter(labels[mask]).items()):
                split_summary.append({"split": split, "grouping": grouping, "group": label, "count": count})
    write_csv(tables / "split_balance.csv", split_summary)
    sets = [set(ids[assignment == name].tolist()) for name in sizes]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    color_values = np.linspace(0, 45, X.shape[1])
    for trajectory in X:
        ax.scatter(trajectory[::4, 0], trajectory[::4, 1], c=color_values[::4], cmap="turbo", s=1.4, alpha=.11)
    ax.set_aspect("equal"); ax.set_xlabel("LAEA x (km)"); ax.set_ylabel("LAEA y (km)")
    ax.set_title("Frozen NOAA cohort: 339 event-aligned 45-day trajectories")
    fig.tight_layout(); fig.savefig(figures / "cohort/frozen_spaghetti.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    colors = {"inference": "#1b9e77", "validation": "#d95f02", "final_test": "#7570b3"}
    for ax, split in zip(axes, sizes, strict=True):
        mask = assignment == split
        ax.scatter(X[mask, 0, 0], X[mask, 0, 1], s=12, alpha=.5, label="day 0", color="#4c78a8")
        ax.scatter(X[mask, -1, 0], X[mask, -1, 1], s=15, alpha=.65, label="day 45", color=colors[split])
        ax.set_title(f"{split}: N={mask.sum()}"); ax.set_aspect("equal"); ax.grid(alpha=.2)
        ax.set_xlabel("x (km)")
    axes[0].set_ylabel("y (km)"); axes[0].legend()
    fig.suptitle("Frozen ID-disjoint split endpoint support")
    fig.tight_layout(); fig.savefig(figures / "splits/endpoints_by_split.png", dpi=180); plt.close(fig)

    labels = ["DJF", "MAM", "JJA", "SON"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bottom = np.zeros(3)
    for label in labels:
        values = [np.sum((assignment == split) & (season == label)) for split in sizes]
        axes[0].bar(list(sizes), values, bottom=bottom, label=label); bottom += values
    axes[0].set_title("Season balance"); axes[0].legend(ncol=2)
    decades = sorted(np.unique(decade))
    width = .24; positions = np.arange(len(decades))
    for j, split in enumerate(sizes):
        axes[1].bar(positions + (j - 1) * width, [np.sum((assignment == split) & (decade == d)) for d in decades], width, label=split)
    axes[1].set_xticks(positions, [f"{d}s" for d in decades]); axes[1].set_title("Decade balance"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "splits/temporal_balance.png", dpi=180); plt.close(fig)

    manifest = {
        "cohort_n": len(X), "shape": list(X.shape), "raw_sha256": sha256(raw_path),
        "cohort_sha256": sha256(processed / "cohort_45d.npz"),
        "development_sha256": sha256(processed / "development_270.npz"),
        "split_manifest_sha256": sha256(processed / "splits/split_manifest.csv"),
        "final_test_ids_sha256": sha256(processed / "splits/final_test_ids.txt"),
        "repeated_cv_manifest_sha256": sha256(processed / "splits/repeated_cv_manifest.csv"),
        "final_test_locked": True, "downstream_artifact_contains_final_test": False,
        "sizes": {"development": 270, **sizes},
        "repeated_cv_folds": int(split_sizes["repeated_cv_folds"]),
        "cv_fold_sizes": {"inference": 200, "validation": 70},
        "domain_km": final.tolist(),
        "exploratory_outside_observation_fraction": float(outside_old.mean()),
        "exploratory_ever_outside_trajectories": int(outside_old.any(axis=1).sum()),
        "final_outside_observation_fraction": float(outside_final.mean()),
    }
    write_json(processed / "cohort_manifest.json", manifest)
    print(
        f"[cohort] frozen {X.shape}; development/final=270/69; "
        f"primary={Counter(assignment)}; cv_folds={split_sizes['repeated_cv_folds']}; final-test locked",
        flush=True,
    )


if __name__ == "__main__":
    main()
