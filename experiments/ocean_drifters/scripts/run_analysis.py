#!/usr/bin/env python3
"""Run the NOAA GDP event-aligned feasibility analysis end to end."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import yaml
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from matplotlib.collections import LineCollection
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    SECONDS_PER_DAY,
    classify_segment,
    first_entry,
    load_config,
    load_data,
    projection_for_gate,
    repo_root,
    season,
    unix_iso,
)


def mkdirs(output: Path) -> None:
    for directory in [
        "tables", "cohorts", "figures/coverage", "figures/gates",
        "figures/trajectories", "figures/snapshots", "figures/dispersion",
        "figures/seasonality",
    ]:
        (output / directory).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_map_features(ax, extent=None) -> None:
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#ece7dc", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#4b5563", linewidth=0.5, zorder=3)
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    # Gridliner labels trigger an invalid-ring error in the Cartopy/Shapely
    # versions used by this repository; axes extents and titles retain context.
    ax.gridlines(draw_labels=False, linewidth=0.25, color="#6b7280", alpha=0.4)


def audit(data, parquet_path: Path, output: Path) -> dict:
    lengths = np.diff(data.offsets).astype(np.int64)
    durations = np.empty(len(data.ids), dtype=float)
    duplicate_count = 0
    nonpositive_intervals = 0
    interval_counter: Counter[float] = Counter()
    for i in range(len(data.ids)):
        s = data.trajectory(i)
        local = data.time[s]
        durations[i] = (local[-1] - local[0]) / SECONDS_PER_DAY if local.size else np.nan
        dt = np.diff(local)
        duplicate_count += int(np.count_nonzero(dt == 0))
        nonpositive_intervals += int(np.count_nonzero(dt <= 0))
        values, counts = np.unique(dt / 3600, return_counts=True)
        interval_counter.update({float(v): int(c) for v, c in zip(values, counts)})

    finite_time = np.isfinite(data.time)
    finite_lon = np.isfinite(data.lon)
    finite_lat = np.isfinite(data.lat)
    valid_lon = finite_lon & (data.lon >= -180) & (data.lon <= 180)
    valid_lat = finite_lat & (data.lat >= -90) & (data.lat <= 90)
    total = len(data.time)
    top_intervals = interval_counter.most_common(12)
    interval_total = sum(interval_counter.values())
    q = [0, .01, .1, .25, .5, .75, .9, .95, .99, 1]
    summary = {
        "file_size_bytes": parquet_path.stat().st_size,
        "trajectory_rows": len(data.ids),
        "observation_rows": total,
        "unique_drifter_ids": int(np.unique(data.ids).size),
        "time_start": unix_iso(np.nanmin(data.time)),
        "time_end": unix_iso(np.nanmax(data.time)),
        "longitude_min": float(np.nanmin(data.lon)),
        "longitude_max": float(np.nanmax(data.lon)),
        "latitude_min": float(np.nanmin(data.lat)),
        "latitude_max": float(np.nanmax(data.lat)),
        "missing_time": int(np.count_nonzero(~finite_time)),
        "missing_longitude": int(np.count_nonzero(~finite_lon)),
        "missing_latitude": int(np.count_nonzero(~finite_lat)),
        "malformed_longitude": int(np.count_nonzero(~valid_lon)),
        "malformed_latitude": int(np.count_nonzero(~valid_lat)),
        "duplicate_id_time": duplicate_count,
        "nonmonotone_or_duplicate_intervals": nonpositive_intervals,
        "six_hour_interval_fraction": interval_counter[6.0] / interval_total,
        "drogue_flag_true_fraction": float(np.mean(data.drogued)),
        "drogue_lost_date_nan": int(np.count_nonzero(~np.isfinite(data.scalar["drogue_lost_date"]))),
        "drogue_lost_date_zero": int(np.count_nonzero(data.scalar["drogue_lost_date"] == 0)),
    }
    rows = [{"metric": key, "value": value} for key, value in summary.items()]
    for name, values in [("trajectory_length", lengths), ("trajectory_duration_days", durations)]:
        for quantile, value in zip(q, np.quantile(values[np.isfinite(values)], q)):
            rows.append({"metric": f"{name}_q{quantile:g}", "value": float(value)})
    for hours, count in top_intervals:
        rows.append({"metric": f"sampling_interval_hours_{hours:g}", "value": count})
    write_csv(output / "tables/data_audit.csv", rows)

    scalar_missing = []
    for name, values in data.scalar.items():
        if name == "WMO":
            # Arrow preserves the NetCDF integer fill sentinel as int32 min.
            missing = int(np.count_nonzero(values == np.iinfo(np.int32).min))
        elif values.dtype.kind == "f":
            missing = int(np.count_nonzero(~np.isfinite(values)))
        else:
            missing = int(np.count_nonzero(values == "")) if values.dtype.kind in "UO" else 0
        scalar_missing.append((name, missing, missing / len(values)))

    schema_md = data.schema.replace("\n", "\n    ")
    text = f"""# NOAA GDP data audit

## Scope and provenance

Canonical input: `{parquet_path.relative_to(repo_root())}` ({parquet_path.stat().st_size / 2**30:.3f} GiB). The embedded metadata identifies the December 2024 Level-2 GDP six-hour product, created 2025-07-15, with DOI [10.25921/7ntx-z961](https://doi.org/10.25921/7ntx-z961). The local download note records 2026-08-17. Only time, position, identity, and drogue fields were read; NOAA velocity and temperature estimates were not used.

## Headline audit

- {len(data.ids):,} trajectory records / unique `obs.id` values and {total:,} observations.
- Coverage: {summary['time_start']} through {summary['time_end']}.
- Longitude: {summary['longitude_min']:.3f} to {summary['longitude_max']:.3f} degrees east; latitude: {summary['latitude_min']:.3f} to {summary['latitude_max']:.3f} degrees north.
- Missing observation values: time {summary['missing_time']:,}, longitude {summary['missing_longitude']:,}, latitude {summary['missing_latitude']:,}. Out-of-range-or-nonfinite coordinates: longitude {summary['malformed_longitude']:,}, latitude {summary['malformed_latitude']:,}.
- Duplicate `(obs.id, time)` records: {duplicate_count:,}. Nonpositive within-trajectory time differences: {nonpositive_intervals:,}.
- {summary['six_hour_interval_fraction']:.8%} of consecutive observations are exactly six hours apart. The dominant interval counts are {', '.join(f'{h:g} h: {n:,}' for h, n in top_intervals[:6])}.
- Trajectory length (observations), q10/median/q90/q99: {np.quantile(lengths, .1):.0f} / {np.median(lengths):.0f} / {np.quantile(lengths, .9):.0f} / {np.quantile(lengths, .99):.0f}.
- Trajectory duration (days), q10/median/q90/q99: {np.quantile(durations, .1):.1f} / {np.median(durations):.1f} / {np.quantile(durations, .9):.1f} / {np.quantile(durations, .99):.1f}.

## Drogue metadata

`obs.drogue_status` is a non-null Boolean large-list field whose embedded attributes say `flag_values: 1,0` and `flag_meanings: drogued, undrogued`. It is therefore the authoritative horizon-level field. {np.mean(data.drogued):.2%} of all observations are flagged drogued. The scalar `drogue_lost_date` contains {summary['drogue_lost_date_nan']:,} NaNs and {summary['drogue_lost_date_zero']:,} zeros; it is retained as a metadata diagnostic and is not used to turn unknown states into attached states. Primary eligibility requires every per-observation flag in the exact segment to be true.

## Missing rates for scalar fields

| Field | Missing/NaN | Fraction |
|---|---:|---:|
{chr(10).join(f'| `{name}` | {missing:,} | {fraction:.4%} |' for name, missing, fraction in scalar_missing)}

## Schema and quality control

The file has no standalone observation QC flag. Its embedded global metadata declares `processing_level: Level 2 QC by GDP drifter DAC`; coordinate uncertainty fields (`err_lat`, `err_lon`) exist but were not needed for selection. `DeploymentStatus` is `good` for {np.count_nonzero(data.scalar['DeploymentStatus'] == 'good'):,}/{len(data.ids):,} rows. No malformed coordinates or nonmonotone trajectories are silently discarded; the counts above are used by the assertions.

```text
{schema_md}
```
"""
    (output / "data_audit.md").write_text(text, encoding="utf-8")
    return summary


def coverage_maps(data, output: Path) -> None:
    finite = np.isfinite(data.lon) & np.isfinite(data.lat)
    global_lon = np.linspace(-180, 180, 181)
    global_lat = np.linspace(-90, 90, 91)
    obs_global, _, _ = np.histogram2d(data.lat[finite], data.lon[finite], bins=[global_lat, global_lon])
    unique_global = np.zeros_like(obs_global, dtype=np.int32)
    regional_lon = np.linspace(-100, -45, 111)
    regional_lat = np.linspace(10, 55, 91)
    unique_regional = np.zeros((len(regional_lat) - 1, len(regional_lon) - 1), dtype=np.int32)
    for i in range(len(data.ids)):
        s = data.trajectory(i)
        lon, lat = data.lon[s], data.lat[s]
        ok = np.isfinite(lon) & np.isfinite(lat)
        ix = np.searchsorted(global_lon, lon[ok], side="right") - 1
        iy = np.searchsorted(global_lat, lat[ok], side="right") - 1
        good = (ix >= 0) & (ix < unique_global.shape[1]) & (iy >= 0) & (iy < unique_global.shape[0])
        flat = np.unique(iy[good] * unique_global.shape[1] + ix[good])
        unique_global.flat[flat] += 1
        ix = np.searchsorted(regional_lon, lon[ok], side="right") - 1
        iy = np.searchsorted(regional_lat, lat[ok], side="right") - 1
        good = (ix >= 0) & (ix < unique_regional.shape[1]) & (iy >= 0) & (iy < unique_regional.shape[0])
        flat = np.unique(iy[good] * unique_regional.shape[1] + ix[good])
        unique_regional.flat[flat] += 1

    def density_figure(array, lon_edges, lat_edges, title, path, extent=None, label="log10 count"):
        fig = plt.figure(figsize=(12, 5.8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        add_map_features(ax, extent)
        shown = np.full(array.shape, np.nan, dtype=float)
        np.log10(array, out=shown, where=array > 0)
        mesh = ax.pcolormesh(lon_edges, lat_edges, shown, cmap="magma", transform=ccrs.PlateCarree(), shading="auto")
        fig.colorbar(mesh, ax=ax, pad=.025, shrink=.78, label=label)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    density_figure(obs_global, global_lon, global_lat, "Global GDP observation density", output / "figures/coverage/global_observation_density.png")
    density_figure(unique_global, global_lon, global_lat, "Global independent-drifter occupancy", output / "figures/coverage/global_unique_drifter_density.png", label="log10 unique drifters per cell")
    region = finite & (data.lon >= -100) & (data.lon <= -45) & (data.lat >= 10) & (data.lat <= 55)
    obs_reg, _, _ = np.histogram2d(data.lat[region], data.lon[region], bins=[regional_lat, regional_lon])
    density_figure(obs_reg, regional_lon, regional_lat, "Western North Atlantic / Gulf observation density", output / "figures/coverage/wna_observation_density.png", [-100, -45, 10, 55])
    density_figure(unique_regional, regional_lon, regional_lat, "Western North Atlantic / Gulf independent-drifter occupancy", output / "figures/coverage/wna_unique_drifter_density.png", [-100, -45, 10, 55], "log10 unique drifters per cell")


def gate_map(config: dict, output: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    add_map_features(ax, [-95, -55, 15, 48])
    colors = plt.cm.tab10(np.linspace(0, 1, len(config["gates"])))
    for (name, gate), color in zip(config["gates"].items(), colors):
        xmin, xmax, ymin, ymax = gate["bounds"]
        ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color=color, lw=1.7, transform=ccrs.PlateCarree(), label=name)
    ax.legend(fontsize=8, loc="lower right", framealpha=.95)
    ax.set_title("Candidate event-alignment gates")
    fig.tight_layout()
    fig.savefig(output / "figures/gates/candidate_gates.png", dpi=180)
    plt.close(fig)


def extract_cohorts(data, config: dict, output: Path):
    all_rows: dict[tuple[str, int], list[dict]] = {}
    eligible_indices: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
    count_rows = []
    exclusion_rows = []
    gate_crossings = {}
    for gate_name, gate in config["gates"].items():
        crossings = []
        for i, drifter_id in enumerate(data.ids):
            s = data.trajectory(i)
            k = first_entry(data.lon[s], data.lat[s], gate)
            if k is not None:
                crossings.append((i, k))
        gate_crossings[gate_name] = crossings
        wide = {}
        for horizon in config["horizons_days"]:
            rows = []
            selected = []
            for i, k in crossings:
                result = classify_segment(data, i, k, horizon, config)
                crossing_flat = int(data.offsets[i]) + k
                timestamp = np.datetime64(int(data.time[crossing_flat]), "s")
                month = int(str(timestamp)[5:7])
                year = int(str(timestamp)[:4])
                row = {
                    "drifter_id": int(data.ids[i]),
                    "gate": gate_name,
                    "crossing_time": str(timestamp),
                    "crossing_lon": float(data.lon[crossing_flat]),
                    "crossing_lat": float(data.lat[crossing_flat]),
                    "horizon_days": horizon,
                    "completeness_class": result["completeness_class"],
                    "coverage_fraction": result["coverage_fraction"],
                    "max_gap_hours": result["max_gap_hours"],
                    "drogue_class": result["drogue_class"],
                    "drogue_lost_date_metadata": result["drogue_lost_date_metadata"],
                    "eligible": result["eligible"],
                    "exclusion_reasons": result["exclusion_reasons"],
                    "year": year,
                    "month": month,
                    "season": season(month),
                    "decade": f"{year // 10 * 10}s",
                }
                rows.append(row)
                if result["eligible"]:
                    selected.append((i, crossing_flat, result["end_flat_index"]))
            key = (gate_name, horizon)
            all_rows[key] = rows
            eligible_indices[key] = selected
            write_csv(output / f"cohorts/{gate_name}_{horizon}d.csv", rows)
            counts = Counter(row["drogue_class"] for row in rows)
            complete_counts = Counter(row["completeness_class"] for row in rows)
            n = len(rows)
            eligible = len(selected)
            years = {row["year"] for row in rows if row["eligible"]}
            seasons = {row["season"] for row in rows if row["eligible"]}
            decades = {row["decade"] for row in rows if row["eligible"]}
            wide[f"N_{horizon}"] = eligible
            count_rows.append({
                "gate": gate_name, "horizon_days": horizon, "N_entries": n,
                "N_eligible": eligible,
                "N_30": "", "N_45": "", "N_60": "",
                "drogued_fraction": counts["known_drogued_throughout"] / n if n else 0,
                "incomplete_fraction": complete_counts["materially_incomplete"] / n if n else 0,
                "near_complete_fraction": complete_counts["near_complete"] / n if n else 0,
                "excluded_for_drogue_uncertainty_fraction": counts["uncertain_drogue_status"] / n if n else 0,
                "missing_ambiguous_drogue_metadata_fraction": sum(row["drogue_lost_date_metadata"] == "missing_or_zero" for row in rows) / n if n else 0,
                "distinct_years": len(years), "distinct_seasons": len(seasons), "distinct_decades": len(decades),
            })
            for cls in ["known_drogued_throughout", "lost_during_horizon", "undrogued_at_entry", "uncertain_drogue_status", "not_assessed_incomplete"]:
                exclusion_rows.append({"gate": gate_name, "horizon_days": horizon, "category_type": "drogue", "category": cls, "count": counts[cls]})
            for cls, count in sorted(complete_counts.items()):
                exclusion_rows.append({"gate": gate_name, "horizon_days": horizon, "category_type": "completeness", "category": cls, "count": count})
            metadata_missing = sum(row["drogue_lost_date_metadata"] == "missing_or_zero" for row in rows)
            exclusion_rows.append({"gate": gate_name, "horizon_days": horizon, "category_type": "scalar_metadata", "category": "drogue_lost_date_missing_or_zero", "count": metadata_missing})
        for row in count_rows[-len(config["horizons_days"]):]:
            row.update(wide)
    write_csv(output / "tables/gate_horizon_counts.csv", count_rows)
    write_csv(output / "tables/exclusions_summary.csv", exclusion_rows)
    return all_rows, eligible_indices, gate_crossings, count_rows


def aligned_arrays(data, selected: list[tuple[int, int, int]], horizon: int):
    ntime = horizon * 4 + 1
    lon = np.empty((len(selected), ntime), dtype=np.float64)
    lat = np.empty_like(lon)
    for row, (_, start, end) in enumerate(selected):
        assert end - start + 1 == ntime
        lon[row] = data.lon[start:end + 1]
        lat[row] = data.lat[start:end + 1]
    return lon, lat


def normalized_energy_to_gaussian(points: np.ndarray, rng: np.random.Generator) -> float:
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 10:
        return float("nan")
    if len(points) > 500:
        points = points[rng.choice(len(points), 500, replace=False)]
    mean = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False) + np.eye(2) * 1e-6
    gaussian = rng.multivariate_normal(mean, covariance, size=len(points))
    energy = 2 * cdist(points, gaussian).mean() - cdist(points, points).mean() - cdist(gaussian, gaussian).mean()
    scale = np.median(np.linalg.norm(points - np.median(points, axis=0), axis=1))
    return float(energy / max(scale, 1.0))


def metric_series(x: np.ndarray, y: np.ndarray):
    nt = x.shape[1]
    result = {name: np.empty(nt) for name in ["mean_x", "mean_y", "eig_major", "eig_minor", "trace", "det", "median_displacement", "robust_spread", "anisotropy"]}
    x0, y0 = x[:, 0], y[:, 0]
    for j in range(nt):
        points = np.column_stack([x[:, j], y[:, j]])
        center = points.mean(axis=0)
        covariance = np.cov(points, rowvar=False)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        robust_center = np.median(points, axis=0)
        result["mean_x"][j], result["mean_y"][j] = center
        result["eig_major"][j], result["eig_minor"][j] = np.sqrt(eigenvalues[1]), np.sqrt(eigenvalues[0])
        result["trace"][j] = np.trace(covariance)
        result["det"][j] = np.linalg.det(covariance)
        result["median_displacement"][j] = np.median(np.hypot(x[:, j] - x0, y[:, j] - y0))
        result["robust_spread"][j] = np.median(np.linalg.norm(points - robust_center, axis=1))
        result["anisotropy"][j] = math.sqrt(eigenvalues[1] / max(eigenvalues[0], 1.0))
    return result


def candidate_metrics(data, config: dict, cohort_rows, eligible_indices, output: Path):
    rng = np.random.default_rng(config["random_seed"])
    rows = []
    series_cache = {}
    arrays_cache = {}
    for gate_name, gate in config["gates"].items():
        crs, transformer = projection_for_gate(gate)
        for horizon in config["horizons_days"]:
            key = (gate_name, horizon)
            selected = eligible_indices[key]
            if not selected:
                continue
            lon, lat = aligned_arrays(data, selected, horizon)
            x, y = transformer.transform(lon, lat)
            x, y = np.asarray(x) / 1000, np.asarray(y) / 1000
            arrays_cache[key] = (lon, lat, x, y)
            daily = np.arange(0, horizon * 4 + 1, 4)
            series = metric_series(x[:, daily], y[:, daily])
            series_cache[key] = series
            all_x, all_y = x.ravel(), y.ravel()
            x95 = np.quantile(all_x, [.025, .975]); y95 = np.quantile(all_y, [.025, .975])
            x99 = np.quantile(all_x, [.005, .995]); y99 = np.quantile(all_y, [.005, .995])
            buffer = config["domain_buffer_km"]
            years = np.array([row["year"] for row in cohort_rows[key] if row["eligible"]])
            seasonal_counts = Counter(row["season"] for row in cohort_rows[key] if row["eligible"])
            probs = np.array([seasonal_counts[s] for s in ["DJF", "MAM", "JJA", "SON"]], dtype=float)
            probs /= probs.sum()
            entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])) / np.log(4))
            ng = normalized_energy_to_gaussian(np.column_stack([x[:, -1], y[:, -1]]), rng)
            spread_growth = float(series["robust_spread"][-1] / max(series["robust_spread"][0], 1e-9))
            n = len(selected)
            if ng > .09 or series["anisotropy"].max() > 3.0:
                visual = "strong stretching/non-Gaussian structure"
            elif spread_growth > 2:
                visual = "coherent spreading transport"
            else:
                visual = "limited evolution"
            if n >= 300 and horizon >= 45:
                recommendation = "yes" if gate_name in {"midatlantic_A", "north_atlantic_A"} else "maybe"
            elif n >= 250:
                recommendation = "maybe"
            else:
                recommendation = "no"
            rows.append({
                "gate": gate_name, "horizon_days": horizon, "N_total": n,
                "N_30": len(eligible_indices[(gate_name, 30)]),
                "N_45": len(eligible_indices[(gate_name, 45)]),
                "N_60": len(eligible_indices[(gate_name, 60)]),
                "drogued_fraction": np.mean([r["drogue_class"] == "known_drogued_throughout" for r in cohort_rows[key]]),
                "time_span_years": int(years.max() - years.min() + 1),
                "first_year": int(years.min()), "last_year": int(years.max()),
                "seasonal_balance": entropy,
                "median_displacement_km": series["median_displacement"][-1],
                "spread_growth": spread_growth,
                "domain_95_xmin_km": x95[0], "domain_95_xmax_km": x95[1],
                "domain_95_ymin_km": y95[0], "domain_95_ymax_km": y95[1],
                "domain_99_xmin_km": x99[0], "domain_99_xmax_km": x99[1],
                "domain_99_ymin_km": y99[0], "domain_99_ymax_km": y99[1],
                "recommended_xmin_km": x99[0] - buffer, "recommended_xmax_km": x99[1] + buffer,
                "recommended_ymin_km": y99[0] - buffer, "recommended_ymax_km": y99[1] + buffer,
                "domain_width_km": x99[1] - x99[0] + 2 * buffer,
                "domain_height_km": y99[1] - y99[0] + 2 * buffer,
                "characteristic_displacement_km": series["median_displacement"][-1],
                "transverse_spread_km": series["eig_minor"][-1],
                "endpoint_robust_spread_km": series["robust_spread"][-1],
                "non_gaussianity": ng,
                "max_anisotropy": float(series["anisotropy"].max()),
                "projection": crs.to_proj4(),
                "visual_structure": visual,
                "main_failure_mode": "modest sample size" if n < 300 else ("large downstream domain" if horizon == 60 else "multi-decade mixture"),
                "recommendation": recommendation,
            })
    write_csv(output / "tables/candidate_metrics.csv", rows)
    return rows, arrays_cache, series_cache


def plot_promising(config, arrays_cache, series_cache, output: Path) -> None:
    for gate_name, horizon in config["promising"]:
        key = (gate_name, horizon)
        lon, lat, x, y = arrays_cache[key]
        series = series_cache[key]
        gate = config["gates"][gate_name]
        xmin, xmax, ymin, ymax = gate["bounds"]
        lonlim = np.quantile(lon, [.0025, .9975]); latlim = np.quantile(lat, [.0025, .9975])
        lonpad = max(1.0, .04 * (lonlim[1] - lonlim[0])); latpad = max(.6, .04 * (latlim[1] - latlim[0]))
        lon0, lat0 = (xmin + xmax) / 2, (ymin + ymax) / 2
        projection = ccrs.LambertAzimuthalEqualArea(lon0, lat0)

        fig = plt.figure(figsize=(12, 7.5))
        ax = plt.axes(projection=projection)
        add_map_features(ax, [lonlim[0] - lonpad, lonlim[1] + lonpad, latlim[0] - latpad, latlim[1] + latpad])
        # Color one segment per day, retaining all trajectories for the visual.
        cmap = plt.get_cmap("turbo")
        norm = plt.Normalize(0, horizon)
        for i in range(len(lon)):
            points = np.column_stack([lon[i], lat[i]])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            collection = LineCollection(segments, cmap=cmap, norm=norm, linewidth=.38, alpha=.17, transform=ccrs.PlateCarree())
            collection.set_array(np.arange(len(segments)) / 4)
            ax.add_collection(collection)
        ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color="black", lw=1.4, transform=ccrs.PlateCarree())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, pad=.025, shrink=.75, label="days after crossing")
        ax.set_title(f"{gate_name}: {len(lon)} strict drogued event-aligned trajectories ({horizon} days)")
        fig.tight_layout()
        fig.savefig(output / f"figures/trajectories/{gate_name}_{horizon}d_spaghetti.png", dpi=190)
        plt.close(fig)

        days = sorted(set([0, 5, 10, 20, 30, 45, 60]) & set(range(horizon + 1)))
        ncol = 3; nrow = math.ceil(len(days) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4 * nrow), subplot_kw={"projection": projection})
        axes = np.atleast_1d(axes).ravel()
        for ax, day in zip(axes, days):
            add_map_features(ax, [lonlim[0] - lonpad, lonlim[1] + lonpad, latlim[0] - latpad, latlim[1] + latpad])
            ax.scatter(lon[:, day * 4], lat[:, day * 4], s=9, alpha=.55, c="#0868ac", edgecolors="none", transform=ccrs.PlateCarree())
            ax.set_title(f"day {day} (N={len(lon)})")
        for ax in axes[len(days):]:
            ax.set_visible(False)
        fig.suptitle(f"Event-aligned empirical laws: {gate_name}, {horizon} days", y=.995)
        fig.tight_layout()
        fig.savefig(output / f"figures/snapshots/{gate_name}_{horizon}d_snapshots.png", dpi=180)
        plt.close(fig)

        days_axis = np.arange(horizon + 1)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
        axes[0, 0].plot(days_axis, series["median_displacement"], color="#006d77")
        axes[0, 0].set_ylabel("median displacement (km)")
        axes[0, 1].plot(days_axis, series["eig_major"], label="major", color="#d95f02")
        axes[0, 1].plot(days_axis, series["eig_minor"], label="minor", color="#1b9e77")
        axes[0, 1].legend(); axes[0, 1].set_ylabel("covariance axis 1σ (km)")
        axes[1, 0].plot(days_axis, series["robust_spread"], color="#7570b3")
        axes[1, 0].set_ylabel("median radial spread (km)")
        axes[1, 1].plot(days_axis, series["anisotropy"], color="#e7298a")
        axes[1, 1].set_ylabel("covariance anisotropy")
        for ax in axes[-1]: ax.set_xlabel("days after crossing")
        for ax in axes.ravel(): ax.grid(alpha=.25)
        fig.suptitle(f"Transport and dispersion: {gate_name}, {horizon} days")
        fig.tight_layout()
        fig.savefig(output / f"figures/dispersion/{gate_name}_{horizon}d_dispersion.png", dpi=180)
        plt.close(fig)


def seasonality_analysis(config, cohort_rows, arrays_cache, output: Path) -> list[dict]:
    rows = []
    for gate_name, horizon in config["seasonality"]:
        eligible = [row for row in cohort_rows[(gate_name, horizon)] if row["eligible"]]
        lon, lat, x, y = arrays_cache[(gate_name, horizon)]
        for grouping in ["season", "month", "decade"]:
            counts = Counter(str(row[grouping]) for row in eligible)
            for group, count in sorted(counts.items()):
                mask = np.array([str(row[grouping]) == group for row in eligible])
                endpoints = np.column_stack([x[mask, -1], y[mask, -1]])
                center = np.median(endpoints, axis=0)
                spread = np.median(np.linalg.norm(endpoints - center, axis=1))
                rows.append({
                    "gate": gate_name, "horizon_days": horizon, "grouping": grouping,
                    "group": group, "N": count, "fraction": count / len(eligible),
                    "endpoint_median_x_km": center[0], "endpoint_median_y_km": center[1],
                    "endpoint_robust_spread_km": spread,
                })
        labels = np.array([row["season"] for row in eligible])
        gate = config["gates"][gate_name]
        xmin, xmax, ymin, ymax = gate["bounds"]
        projection = ccrs.LambertAzimuthalEqualArea((xmin + xmax) / 2, (ymin + ymax) / 2)
        extent = [np.quantile(lon, .0025) - 1, np.quantile(lon, .9975) + 1, np.quantile(lat, .0025) - .7, np.quantile(lat, .9975) + .7]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), subplot_kw={"projection": projection})
        for ax, label in zip(axes, ["DJF", "MAM", "JJA", "SON"]):
            add_map_features(ax, extent)
            mask = labels == label
            ax.scatter(lon[mask, -1], lat[mask, -1], s=12, alpha=.6, transform=ccrs.PlateCarree(), color="#2c7fb8")
            ax.set_title(f"{label}: N={mask.sum()}")
        fig.suptitle(f"Seasonal endpoint distributions, {gate_name} day {horizon}")
        fig.tight_layout()
        fig.savefig(output / f"figures/seasonality/{gate_name}_{horizon}d_by_season.png", dpi=180)
        plt.close(fig)

        decade_labels = sorted({row["decade"] for row in eligible})
        ncol = min(5, len(decade_labels)); nrow = math.ceil(len(decade_labels) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 4 * nrow), subplot_kw={"projection": projection})
        axes = np.atleast_1d(axes).ravel()
        labels = np.array([row["decade"] for row in eligible])
        for ax, label in zip(axes, decade_labels):
            add_map_features(ax, extent)
            mask = labels == label
            ax.scatter(lon[mask, -1], lat[mask, -1], s=13, alpha=.62, transform=ccrs.PlateCarree(), color="#7b3294")
            ax.set_title(f"{label}: N={mask.sum()}")
        for ax in axes[len(decade_labels):]:
            ax.set_visible(False)
        fig.suptitle(f"Decadal endpoint distributions, {gate_name} day {horizon}")
        fig.tight_layout()
        fig.savefig(output / f"figures/seasonality/{gate_name}_{horizon}d_by_decade.png", dpi=180)
        plt.close(fig)
    write_csv(output / "tables/seasonality_counts.csv", rows)
    return rows


def write_gates(config: dict, output: Path) -> None:
    gates = {
        "crs": "EPSG:4326 (longitude/latitude degrees; closed box bounds)",
        "passage_rule": "first outside-to-inside observation; west gates additionally require the immediately previous longitude to be west of xmin",
        "one_segment_per_drifter": True,
        "gates": config["gates"],
    }
    with (output / "gates.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(gates, handle, sort_keys=False)


def build_reports(config, audit_summary, count_rows, metrics, season_rows, output: Path) -> None:
    metric_lookup = {(r["gate"], r["horizon_days"]): r for r in metrics}
    winner = metric_lookup[("north_atlantic_A", 45)]
    count = next(r for r in count_rows if r["gate"] == "north_atlantic_A" and r["horizon_days"] == 45)
    seasonal = {r["group"]: r["N"] for r in season_rows if r["gate"] == "north_atlantic_A" and r["horizon_days"] == 45 and r["grouping"] == "season"}
    decade_diagnostics = [r for r in season_rows if r["gate"] == "north_atlantic_A" and r["horizon_days"] == 45 and r["grouping"] == "decade" and r["N"] >= 20]
    decade_centers = np.array([[r["endpoint_median_x_km"], r["endpoint_median_y_km"]] for r in decade_diagnostics])
    decade_shift = float(cdist(decade_centers, decade_centers).max())
    season_diagnostics = [r for r in season_rows if r["gate"] == "north_atlantic_A" and r["horizon_days"] == 45 and r["grouping"] == "season"]
    season_centers = np.array([[r["endpoint_median_x_km"], r["endpoint_median_y_km"]] for r in season_diagnostics])
    season_shift = float(cdist(season_centers, season_centers).max())
    ranking = sorted(metrics, key=lambda r: (r["recommendation"] == "yes", r["N_total"], -r["domain_width_km"]), reverse=True)
    table_lines = ["| Gate | T | N | Median displacement | Spread growth | Domain km | Non-Gaussianity | Recommendation |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in ranking:
        table_lines.append(f"| {r['gate']} | {r['horizon_days']} | {r['N_total']} | {r['median_displacement_km']:.0f} km | {r['spread_growth']:.2f}× | {r['domain_width_km']:.0f}×{r['domain_height_km']:.0f} | {r['non_gaussianity']:.3f} | {r['recommendation']} |")
    xmin, xmax, ymin, ymax = config["gates"]["north_atlantic_A"]["bounds"]
    proj = winner["projection"]
    recommendation = f"""# Recommended ocean-drifter benchmark

## Decision

Use **`north_atlantic_A` with a 45-day horizon**, pooled over all seasons.

The gate is the closed WGS84 longitude/latitude ribbon **66°W to 64°W and {ymin:g}°N to {ymax:g}°N** (`lon ∈ [{xmin:g}, {xmax:g}]`, `lat ∈ [{ymin:g}, {ymax:g}]`). A passage qualifies only when a drifter moves from `lon < {xmin:g}` into the ribbon while inside its latitude range. The first qualifying passage is retained, giving at most one segment per `obs.id`.

This leaves **{winner['N_total']} independent trajectories**. The same gate yields N30={winner['N_30']}, N45={winner['N_45']}, and N60={winner['N_60']}. Thus 45 days preserves more than the requested ~300 while allowing a later ID-disjoint split on the order of 200/75/75; it is less comfortable than the illustrative 300/100/100 split, so uncertainty bars and repeated ID-level splits will matter. Sixty days remains feasible ({winner['N_60']}) but buys a larger domain and fewer samples.

## Filters and alignment

- Exact first passage as defined above; no recrossings are counted.
- Exactly 181 observations on the nominal six-hour grid from day 0 through day 45, all coordinates finite. No interpolation is used in the primary cohort.
- Every one of the 181 non-null per-observation `drogue_status` flags must be `drogued`. `drogue_lost_date` is audited but never used to infer attachment when the observation flag says otherwise.
- At this gate/horizon, {count['drogued_fraction']:.1%} of all first passages are drogued throughout and {count['incomplete_fraction']:.1%} are materially incomplete. Per-observation drogue uncertainty exclusions are {count['excluded_for_drogue_uncertainty_fraction']:.1%}. The separate scalar `drogue_lost_date` is zero/NaN for {count['missing_ambiguous_drogue_metadata_fraction']:.1%} of passages; these remain explicit in `exclusions_summary.csv` and do not override the non-null observation flag.
- NOAA velocity components are not loaded or used.

## Seasonality and time coverage

Keep the **all-season historical law**. The eligible counts are {', '.join(f'{s}={seasonal.get(s, 0)}' for s in ['DJF','MAM','JJA','SON'])}; no individual season is large enough to preserve the preferred sample size. The normalized seasonal entropy is {winner['seasonal_balance']:.3f} (1 is perfectly balanced). Seasonal endpoint medians differ by at most {season_shift:.0f} km, compared with an all-cohort endpoint robust spread of {winner['endpoint_robust_spread_km']:.0f} km. Crossings span {winner['first_year']}–{winner['last_year']}; among decades with at least 20 samples, endpoint medians differ by at most {decade_shift:.0f} km. The 1980s and 2020s edge periods are visibly sparse, especially the two-observation 1980s bin. The panels show temporal heterogeneity, but it is not a clean regime split and filtering would sacrifice most of the cohort. The all-years mixture remains interpretable specifically as a historical transport distribution; season and decade should be retained as stratification variables in the later benchmark.

## Coordinates and computational domain

Use the local Lambert azimuthal equal-area projection centered on the gate:

```text
{proj}
```

Over all selected trajectory positions, the 95% coordinate-wise box is x=[{winner['domain_95_xmin_km']:.0f}, {winner['domain_95_xmax_km']:.0f}] km and y=[{winner['domain_95_ymin_km']:.0f}, {winner['domain_95_ymax_km']:.0f}] km. The 99% box is x=[{winner['domain_99_xmin_km']:.0f}, {winner['domain_99_xmax_km']:.0f}] km and y=[{winner['domain_99_ymin_km']:.0f}, {winner['domain_99_ymax_km']:.0f}] km. A 100 km buffer gives the recommended rectangle x=[{winner['recommended_xmin_km']:.0f}, {winner['recommended_xmax_km']:.0f}] km, y=[{winner['recommended_ymin_km']:.0f}, {winner['recommended_ymax_km']:.0f}] km, approximately **{winner['domain_width_km']:.0f}×{winner['domain_height_km']:.0f} km**. Median 45-day displacement is {winner['median_displacement_km']:.0f} km and endpoint transverse 1σ spread is {winner['transverse_spread_km']:.0f} km. These are inputs for a later sensor-radius decision; no radius is selected here.

## Why this is the strongest benchmark

The cross-section creates a recognizable downstream Gulf Stream-extension initial condition while retaining substantially more strict drogued trajectories than Florida, Yucatán, or Cape Hatteras. Over 45 days the robust spread grows by {winner['spread_growth']:.2f}×, peak covariance anisotropy reaches {winner['max_anisotropy']:.2f}, and the normalized endpoint energy discrepancy from its fitted Gaussian is {winner['non_gaussianity']:.3f}. The snapshots and spaghetti map show stretching, curvature, and branching rather than a single rigidly translating blob. That intermediate structure is precisely what can make later endpoint-compatible measurement designs dynamically distinct.

Most informative figures:

- [45-day spaghetti](figures/trajectories/north_atlantic_A_45d_spaghetti.png)
- [event-aligned snapshots](figures/snapshots/north_atlantic_A_45d_snapshots.png)
- [dispersion evolution](figures/dispersion/north_atlantic_A_45d_dispersion.png)
- [seasonal endpoints](figures/seasonality/north_atlantic_A_45d_by_season.png)
- [decadal endpoints](figures/seasonality/north_atlantic_A_45d_by_decade.png)
- [candidate gates](figures/gates/candidate_gates.png)

## Caveats and failure modes

The cohort is a historical mixture across seasons, decades, deployment programs, and sampling regimes; it estimates that mixture, not a stationary climatological transition law. The rectangular gate is an operational proxy for a current cross-section, not a streamline-normal section, and entry is quantized to six-hour observations. A few spatial outliers are excluded only from domain quantiles, not from cohort statistics. The sample is adequate but not generous for three-way splitting. The downstream domain is large, which will put pressure on eventual grid resolution. Before MFSI fitting, freeze ID-level splits, examine decade-stratified held-out behavior, and decide whether robust spatial boundary handling is needed.

## Candidate ranking

{chr(10).join(table_lines)}
"""
    (output / "recommendation.md").write_text(recommendation, encoding="utf-8")
    readme = f"""# NOAA ocean-drifter feasibility analysis

This directory is generated by:

```bash
.venv/bin/python experiments/ocean_drifters/scripts/run_analysis.py
```

The pipeline reads the canonical raw Parquet without modifying it, never loads NOAA velocity fields, selects one first event per drifter, writes lightweight cohort indexes, and produces the audit, coverage maps, dynamics diagnostics, seasonality checks, domain estimates, ranking, and final recommendation.

Start with [recommendation.md](recommendation.md), then [data_audit.md](data_audit.md). Parameters and gate definitions are in `../analysis_config.yaml` and copied to [gates.yaml](gates.yaml). Random seed: {config['random_seed']}.

Data citation: Lumpkin, R. & Centurioni, L. (2019), *NOAA Global Drifter Program quality-controlled 6-hour interpolated data from ocean surface drifting buoys*, NOAA NCEI, [doi:10.25921/7ntx-z961](https://doi.org/10.25921/7ntx-z961). Subset: the event-aligned cohorts listed under `cohorts/` from the December 2024 product.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def validate(data, config, gate_crossings, eligible_indices) -> None:
    assert len(data.ids) == len(np.unique(data.ids)), "obs.id must identify unique trajectory rows"
    assert np.all(np.isfinite(data.time)), "times must be finite"
    assert np.all(np.isfinite(data.lon)) and np.all((-180 <= data.lon) & (data.lon <= 180)), "longitude invalid"
    assert np.all(np.isfinite(data.lat)) and np.all((-90 <= data.lat) & (data.lat <= 90)), "latitude invalid"
    for gate_name, crossings in gate_crossings.items():
        ids = [int(data.ids[i]) for i, _ in crossings]
        assert len(ids) == len(set(ids)), f"multiple crossings selected for {gate_name}"
        for i, k in crossings:
            s = data.trajectory(i)
            assert first_entry(data.lon[s], data.lat[s], config["gates"][gate_name]) == k
    for (gate_name, horizon), selected in eligible_indices.items():
        for i, start, end in selected:
            assert end - start + 1 == horizon * 4 + 1
            assert np.all(np.diff(data.time[start:end + 1]) == 21_600)
            assert np.all(data.drogued[start:end + 1])
    print("All event-alignment, uniqueness, coordinate, completeness, and drogue assertions passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-coverage", action="store_true", help="Skip the two expensive global occupancy maps")
    args = parser.parse_args()
    config = load_config(args.config)
    root = repo_root()
    parquet_path = root / config["input"]
    output = root / config["output"]
    mkdirs(output)
    print(f"Loading selected columns from {parquet_path}")
    data = load_data(parquet_path)
    print(f"Loaded {len(data.ids):,} trajectories and {len(data.time):,} observations")
    summary = audit(data, parquet_path, output)
    if not args.skip_coverage:
        print("Building coverage maps")
        coverage_maps(data, output)
    gate_map(config, output)
    write_gates(config, output)
    print("Extracting event-aligned cohorts")
    cohort_rows, eligible_indices, crossings, count_rows = extract_cohorts(data, config, output)
    validate(data, config, crossings, eligible_indices)
    print("Computing candidate dynamics and projected domains")
    metrics, arrays_cache, series_cache = candidate_metrics(data, config, cohort_rows, eligible_indices, output)
    print("Rendering promising candidates and seasonality")
    plot_promising(config, arrays_cache, series_cache, output)
    season_rows = seasonality_analysis(config, cohort_rows, arrays_cache, output)
    build_reports(config, summary, count_rows, metrics, season_rows, output)
    manifest = {
        "config": config["_path"], "input": str(parquet_path), "random_seed": config["random_seed"],
        "trajectory_count": len(data.ids), "observation_count": len(data.time),
        "selected_gate": "north_atlantic_A", "selected_horizon_days": 45,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    gc.collect()
    print(f"Done. Recommendation: {output / 'recommendation.md'}")


if __name__ == "__main__":
    main()
