from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np


def jsonable(value: Any) -> Any:
    """Convert JAX/NumPy/dataclass values only at the serialization boundary."""
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, jax.Array):
        arr = np.asarray(value)
        return arr.item() if arr.ndim == 0 else arr.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_result_bundle(json_path: str | Path) -> dict[str, Any]:
    """Load a result JSON plus known companion CSVs when they exist.

    Legacy field names are preserved instead of silently rewriting values. A UI
    adapter may normalize display fields without changing the underlying data.
    """
    json_path = Path(json_path)
    bundle: dict[str, Any] = {"result": read_json(json_path)}

    companions = {
        "candidate_summary": json_path.with_suffix(".candidate_summary.csv"),
        "validation_trials": json_path.with_suffix(".validation_trials.csv"),
    }
    for name, path in companions.items():
        if path.exists():
            bundle[name] = read_csv(path)

    return bundle


def write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key not in seen and not isinstance(value, (dict, list, tuple)):
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k, "")) for k in keys})
    return path
