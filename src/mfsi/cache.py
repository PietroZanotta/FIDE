from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _atomic_replace(tmp: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _atomic_replace(tmp, path)
    return path


def save_npz_cache(
    path: str | Path,
    arrays: Mapping[str, Any],
    *,
    signature: str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write an NPZ cache carrying its compatibility signature."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["__signature__"] = np.asarray(signature)
    payload["__metadata_json__"] = np.asarray(
        json.dumps(dict(metadata or {}), sort_keys=True, default=_json_default)
    )
    np.savez_compressed(tmp, **payload)
    _atomic_replace(tmp, path)
    return path


def load_npz_cache(
    path: str | Path,
    *,
    signature: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            got = str(np.asarray(data["__signature__"]).item())
            if got != signature:
                return None
            metadata = json.loads(str(np.asarray(data["__metadata_json__"]).item()))
            arrays = {
                key: np.asarray(data[key])
                for key in data.files
                if not key.startswith("__")
            }
            return arrays, metadata
    except Exception:
        return None


def stage_cache_path(output_dir: str | Path, stage: str) -> Path:
    return Path(output_dir) / "cache" / f"{stage}.json"


def load_stage_result(
    output_dir: str | Path,
    stage: str,
    *,
    signature: str,
) -> dict[str, Any] | None:
    path = stage_cache_path(output_dir, stage)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("signature") != signature:
        return None
    return payload.get("result")


def save_stage_result(
    output_dir: str | Path,
    stage: str,
    *,
    signature: str,
    result: Mapping[str, Any],
) -> Path:
    return write_json_atomic(
        stage_cache_path(output_dir, stage),
        {"signature": signature, "result": dict(result)},
    )
