from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
VORTICES_DIR = SCRIPT_DIR.parent / "vortices_percentage"
for _path in (SRC_DIR, VORTICES_DIR, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mfsi.cache import fingerprint, write_json_atomic  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "name", "mode", "seed", "truth", "measurement"}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"configuration is missing: {', '.join(missing)}")
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    return fingerprint(cfg)


def artifact_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "root": root,
        "endpoint": root / "endpoint_reference",
        "prospective": root / "prospective",
        "hidden": root / "hidden_validation",
        "results": root / "results",
    }


def nested_indices(time_n: int, acquisition_n: int) -> np.ndarray:
    if not 2 <= int(acquisition_n) <= int(time_n):
        raise ValueError("acquisition_nodes must lie between 2 and scientific_nodes")
    raw = np.rint(np.linspace(0, time_n - 1, acquisition_n)).astype(np.int32)
    raw[0], raw[-1] = 0, time_n - 1
    if len(np.unique(raw)) != acquisition_n:
        raise ValueError("configuration does not produce unique acquisition nodes")
    return raw


def trap_weights(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=np.float64)
    w = np.zeros_like(t)
    w[0] = 0.5 * (t[1] - t[0])
    w[-1] = 0.5 * (t[-1] - t[-2])
    if len(t) > 2:
        w[1:-1] = 0.5 * (t[2:] - t[:-2])
    return w / np.sum(w)


def geometry_centers(eta: np.ndarray) -> np.ndarray:
    values = np.asarray(eta, dtype=np.float64)
    return values.reshape((-1, 2))


def geometry_valid(eta: np.ndarray, cfg: dict[str, Any]) -> bool:
    centers = geometry_centers(eta)
    m = cfg["measurement"]
    margin = float(m["boundary_margin"])
    in_box = np.all(
        (centers[:, 0] >= margin)
        & (centers[:, 0] <= 2.0 - margin)
        & (centers[:, 1] >= margin)
        & (centers[:, 1] <= 1.0 - margin)
    )
    delta = centers[:, None, :] - centers[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=-1))
    distance += np.eye(len(centers)) * 1.0e9
    return bool(in_box and np.min(distance) >= float(m["min_separation"]))


def software_metadata() -> dict[str, Any]:
    import jax
    import scipy

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        commit = None
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "jax": jax.__version__,
        "scipy": scipy.__version__,
        "git_commit": commit,
    }


def experiment_source_hash() -> str:
    """Hash runtime experiment sources so caches never cross code revisions."""
    from mfsi.cache import file_sha256

    runtime_files = (
        "aggregate_qois.py",
        "build_prospective_data.py",
        "common.py",
        "evaluator.py",
        "physical.py",
        "prospective_data.py",
        "select.py",
        "train_reference.py",
        "validate.py",
    )
    return fingerprint({name: file_sha256(SCRIPT_DIR / name) for name in runtime_files})


__all__ = [
    "SCRIPT_DIR",
    "REPO_ROOT",
    "artifact_dirs",
    "config_hash",
    "experiment_source_hash",
    "fingerprint",
    "geometry_centers",
    "geometry_valid",
    "load_config",
    "nested_indices",
    "software_metadata",
    "trap_weights",
    "write_json_atomic",
]
