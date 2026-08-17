"""Shared, memory-conscious utilities for the NOAA GDP feasibility analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from pyproj import CRS, Transformer

SECONDS_PER_DAY = 86_400


@dataclass
class RaggedData:
    ids: np.ndarray
    offsets: np.ndarray
    time: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    drogued: np.ndarray
    scalar: dict[str, np.ndarray]
    schema: str
    attrs: dict

    def trajectory(self, index: int) -> slice:
        return slice(int(self.offsets[index]), int(self.offsets[index + 1]))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "experiments/ocean_drifters/analysis_config.yaml"
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_path"] = str(path)
    return config


def _values(table, name: str) -> np.ndarray:
    return table[name].combine_chunks().values.to_numpy(zero_copy_only=False)


def load_data(parquet_path: str | Path) -> RaggedData:
    """Load only coordinates, time, drogue flags, and required scalar metadata.

    The source has one Parquet row per trajectory and large-list observation
    fields.  Arrow decompresses the selected columns directly to contiguous
    arrays; velocity and temperature fields are intentionally never loaded.
    """
    parquet_path = Path(parquet_path)
    pf = pq.ParquetFile(parquet_path)
    obs = pq.read_table(
        parquet_path,
        columns=["obs.id", "obs.time", "obs.lon", "obs.lat", "obs.drogue_status"],
        memory_map=True,
    )
    scalar_names = [
        "rowsize", "WMO", "deploy_date", "start_date", "end_date",
        "drogue_lost_date", "DeploymentStatus", "DrogueDetectSensor",
        "typedeath",
    ]
    scalars = pq.read_table(parquet_path, columns=scalar_names, memory_map=True)
    parameters = json.loads((pf.metadata.metadata or {})[b"ak:parameters"])
    attrs = parameters[2]["RecordArray"]["attrs"]
    scalar = {
        name: scalars[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in scalar_names
    }
    times = obs["time"].combine_chunks()
    return RaggedData(
        ids=obs["id"].combine_chunks().to_numpy(zero_copy_only=False),
        offsets=times.offsets.to_numpy(zero_copy_only=False),
        time=_values(obs, "time"),
        lon=_values(obs, "lon"),
        lat=_values(obs, "lat"),
        drogued=_values(obs, "drogue_status").astype(bool, copy=False),
        scalar=scalar,
        schema=str(pf.schema_arrow),
        attrs=attrs,
    )


def first_entry(local_lon: np.ndarray, local_lat: np.ndarray, gate: dict) -> int | None:
    xmin, xmax, ymin, ymax = gate["bounds"]
    valid = np.isfinite(local_lon) & np.isfinite(local_lat)
    inside = valid & (local_lon >= xmin) & (local_lon <= xmax) & (local_lat >= ymin) & (local_lat <= ymax)
    entries = np.flatnonzero(inside & np.r_[False, ~inside[:-1]])
    if gate.get("entry_from", "any") == "west":
        entries = entries[(entries > 0) & (local_lon[entries - 1] < xmin)]
    if not entries.size:
        return None
    return int(entries[0])


def classify_segment(data: RaggedData, trajectory_index: int, start_index: int, horizon: int, config: dict) -> dict:
    segment = data.trajectory(trajectory_index)
    a, b = int(segment.start), int(segment.stop)
    k = a + start_index
    target = data.time[k] + horizon * SECONDS_PER_DAY
    relative_end = int(np.searchsorted(data.time[a:b], target, side="left"))
    end = a + relative_end
    endpoint_present = end < b and data.time[end] == target
    expected = horizon * (24 // int(config["sampling_interval_hours"])) + 1
    if endpoint_present:
        s = slice(k, end + 1)
        dt = np.diff(data.time[s])
        finite_coordinates = bool(np.all(np.isfinite(data.lon[s])) and np.all(np.isfinite(data.lat[s])))
        exact = bool((end - k + 1 == expected) and finite_coordinates and np.all(dt == 21_600))
        coverage = min(1.0, (end - k + 1) / expected)
        max_gap_hours = float(dt.max() / 3600) if dt.size else 0.0
        near = bool(
            not exact
            and finite_coordinates
            and coverage >= config["completeness"]["near_complete_min_fraction"]
            and max_gap_hours <= config["completeness"]["near_complete_max_gap_hours"]
        )
        status = data.drogued[s]
        if bool(np.all(status)):
            drogue_class = "known_drogued_throughout"
        elif bool(status[0]) and bool(np.any(~status)):
            drogue_class = "lost_during_horizon"
        else:
            drogue_class = "undrogued_at_entry"
    else:
        exact = near = False
        coverage = max(0.0, min(1.0, (b - k) / expected))
        max_gap_hours = np.nan
        finite_coordinates = False
        drogue_class = "not_assessed_incomplete"
        end = min(end, b - 1)

    lost_date = float(data.scalar["drogue_lost_date"][trajectory_index])
    lost_date_metadata = "dated" if np.isfinite(lost_date) and lost_date > 0 else "missing_or_zero"
    completeness = "exact_6_hour" if exact else ("near_complete" if near else "materially_incomplete")
    eligible = exact and drogue_class == "known_drogued_throughout"
    reasons = []
    if not exact:
        reasons.append("not_exact_6_hour")
    if drogue_class != "known_drogued_throughout":
        reasons.append(drogue_class)
    return {
        "end_flat_index": end,
        "completeness_class": completeness,
        "coverage_fraction": coverage,
        "max_gap_hours": max_gap_hours,
        "finite_coordinates": finite_coordinates,
        "drogue_class": drogue_class,
        "drogue_lost_date_metadata": lost_date_metadata,
        "eligible": eligible,
        "exclusion_reasons": ";".join(reasons),
    }


def projection_for_gate(gate: dict) -> tuple[CRS, Transformer]:
    xmin, xmax, ymin, ymax = gate["bounds"]
    lon0, lat0 = (xmin + xmax) / 2, (ymin + ymax) / 2
    crs = CRS.from_proj4(
        f"+proj=laea +lat_0={lat0:.6f} +lon_0={lon0:.6f} +datum=WGS84 +units=m +no_defs"
    )
    return crs, Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def unix_iso(seconds: float) -> str:
    if not np.isfinite(seconds):
        return ""
    return str(np.datetime64(int(seconds), "s"))


def season(month: int) -> str:
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[month]
