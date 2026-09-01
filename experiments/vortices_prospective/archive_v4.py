from __future__ import annotations

"""Seal the completed prospective-v4 run in a hash-verified archive."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tarfile
from typing import Any

from common import REPO_ROOT, write_json_atomic
from mfsi.cache import file_sha256


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"archive source contains a symlink: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return rows


def archive_v4(source: str | Path, archive_dir: str | Path) -> dict[str, Any]:
    source = Path(source).resolve()
    archive_dir = Path(archive_dir).resolve()
    manifest_path = source / "results" / "frozen_manifest.json"
    validation_path = source / "results" / "validation_result.json"
    if not manifest_path.exists() or not validation_path.exists():
        raise RuntimeError("v4 must be frozen and validated before archival")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    manifest_sha = file_sha256(manifest_path)
    if manifest.get("status") != "frozen_before_hidden_validation":
        raise RuntimeError("v4 manifest does not have the sealed prevalidation status")
    if validation.get("frozen_manifest_sha256") != manifest_sha:
        raise RuntimeError("v4 validation is not bound to its frozen manifest")
    if not validation.get("fresh_hidden_validation", {}).get("created_after_freeze"):
        raise RuntimeError("v4 validation bank is not certified post-freeze")
    if validation["fresh_hidden_validation"].get("previous_hidden_bank_reused"):
        raise RuntimeError("v4 validation receipt reports hidden-bank reuse")

    archive_dir.mkdir(parents=True, exist_ok=True)
    stem = source.name
    archive_path = archive_dir / f"{stem}.tar.gz"
    receipt_path = archive_dir / f"{stem}.archive.json"
    if archive_path.exists() or receipt_path.exists():
        if not archive_path.exists() or not receipt_path.exists():
            raise RuntimeError("incomplete existing v4 archive; refusing to overwrite")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("archive_sha256") == file_sha256(archive_path)
            and receipt.get("frozen_manifest_sha256") == manifest_sha
            and receipt.get("source_inventory") == _inventory(source)
        ):
            return receipt
        raise RuntimeError("existing v4 archive does not match the sealed source")

    inventory = _inventory(source)
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=9) as handle:
        handle.add(source, arcname=stem, recursive=True)
    os.replace(temporary, archive_path)
    receipt = {
        "schema_version": 1,
        "role": "immutable_historical_archive_of_completed_prospective_v4",
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_relative_to_repo": source.relative_to(REPO_ROOT).as_posix(),
        "source_file_count": len(inventory),
        "source_bytes": sum(row["bytes"] for row in inventory),
        "source_inventory": inventory,
        "frozen_manifest_sha256": manifest_sha,
        "validation_result_sha256": file_sha256(validation_path),
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": file_sha256(archive_path),
        "sealed": True,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "prospective_v4_robust_full",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "archives",
    )
    args = parser.parse_args()
    receipt = archive_v4(args.source, args.archive_dir)
    print(json.dumps({
        "archive": receipt["archive"],
        "archive_sha256": receipt["archive_sha256"],
        "files": receipt["source_file_count"],
        "sealed": receipt["sealed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
