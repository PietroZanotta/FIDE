#!/usr/bin/env python3
"""Render paper-style snapshots for the completed V2.1 0.5--2% designs.

This is deterministic visualization-only post-processing.  It reuses the
established Vortices paper renderer, the frozen V1 truth population, one of
the three qualified V2 references, and trial 0 of the independent C3-64
holdout bank.  It never runs selection or validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEGACY_DIR = HERE
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import visualize_paper as paper  # noqa: E402


PUBLISHED = HERE / "outputs" / "published"
DEFAULT_CONFIG = HERE / "base_experiment_config.json"
DEFAULT_TRUTH_BANK = HERE / "inputs" / "truth_bank.npz"
DEFAULT_REFERENCE_BANK = HERE / "inputs" / "visualization_reference_bank.npz"
DEFAULT_VALIDATION_BANK = HERE / "inputs" / "visualization_holdout_bank.npz"
DEFAULT_OUTPUT_DIR = HERE / "plots"
PAUSE_RECEIPT = PUBLISHED / "selection_pause_receipt.json"
BANK_RECEIPT = PUBLISHED / "holdout_bank_receipt.json"
REFERENCE_RECEIPT = PUBLISHED / "reference_qualification_receipt.json"
GEOMETRY_RECEIPT = PUBLISHED / "selection_geometries.json"
ALLOWANCES = ((0.5, "0p5"), (1.0, "1p0"), (2.0, "2p0"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--truth-bank", type=Path, default=DEFAULT_TRUTH_BANK)
    parser.add_argument("--reference-bank", type=Path, default=DEFAULT_REFERENCE_BANK)
    parser.add_argument("--validation-bank", type=Path, default=DEFAULT_VALIDATION_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def verify_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    pause = load_json(PAUSE_RECEIPT)
    if pause.get("status") != "PAUSED_BY_USER_AFTER_2PCT_PARTIAL_PARETO":
        raise RuntimeError("V2.1 must remain paused after the completed 2% point")
    if pause.get("resume_authorization_present") is not False:
        raise RuntimeError("unexpected resume authorization in pause receipt")
    bank = load_json(BANK_RECEIPT)
    if bank.get("status") != "FROZEN_SHARED_C3_64_CONFIRMATORY_BANK":
        raise RuntimeError("the frozen independent holdout-bank receipt is required")
    if int(bank.get("namespace", -1)) != 23 or int(bank.get("trials", -1)) != 64:
        raise RuntimeError("unexpected holdout-bank namespace or trial count")
    if sha256_file(args.validation_bank) != bank["bank_sha256"]:
        raise RuntimeError("holdout-bank hash does not match its published receipt")
    reference = load_json(REFERENCE_RECEIPT)
    if reference.get("status") != "PASS" or not reference.get("qualified"):
        raise RuntimeError("the published reference is not qualified")
    if sha256_file(args.reference_bank) != reference["rollout_bank_sha256"]:
        raise RuntimeError("reference-bank hash does not match its qualification receipt")
    geometries = load_json(GEOMETRY_RECEIPT)
    if geometries.get("status") != "FROZEN_VISUALIZATION_GEOMETRY_ADAPTER":
        raise RuntimeError("published visualization geometries are not frozen")
    rows = geometries.get("rows", [])
    if [float(row["allowance_percent"]) for row in rows] != [0.5, 1.0, 2.0]:
        raise RuntimeError("published visualization geometries are incomplete")
    if any(len(row.get("full_centers", [])) != 4 for row in rows):
        raise RuntimeError("a published Full geometry does not contain four centers")
    return rows


def write_geometry_adapter(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_dir / "completed_full_geometries_0p5_to_2pct.json"
    atomic_json(
        path,
        {
            "schema_version": 1,
            "status": "FROZEN_VISUALIZATION_GEOMETRY_ADAPTER",
            "data_role": "INDEPENDENT_HOLDOUT_VISUALIZATION_ONLY",
            "rows": rows,
        },
    )
    return path


def relabel_figure(figure: plt.Figure, allowance: float, trial: int) -> None:
    old = (
        f"Authoritative Full geometry at {allowance:g}% allowance"
        f"  ·  frozen validation trial {trial}"
    )
    new = (
        f"V2.1 Full geometry at {allowance:g}% allowance"
        f"  ·  independent holdout trial {trial}"
    )
    replacements = 0
    for text_artist in figure.texts:
        if text_artist.get_text() == old:
            text_artist.set_text(new)
            replacements += 1
    if replacements != 1:
        raise RuntimeError(f"expected one provenance subtitle, replaced {replacements}")


def render_static_set(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = verify_inputs(args)
    adapter = write_geometry_adapter(output_dir, rows)
    artifacts: list[dict[str, Any]] = []

    for allowance, tag in ALLOWANCES:
        render_args = SimpleNamespace(
            config=args.config,
            pareto=adapter,
            truth_bank=args.truth_bank,
            reference_bank=args.reference_bank,
            validation_bank=args.validation_bank,
            allowance=allowance,
            trial=args.trial,
        )
        data = paper._prepare_data(render_args)
        figure = paper.make_figure(data)
        relabel_figure(figure, allowance, args.trial)
        stem = output_dir / f"vortices_v2_1_full_{tag}_paper"
        png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
        figure.savefig(png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
        figure.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
        plt.close(figure)
        artifacts.append(
            {
                "allowance_percent": allowance,
                "png": str(png.relative_to(REPO)),
                "png_sha256": sha256_file(png),
                "pdf": str(pdf.relative_to(REPO)),
                "pdf_sha256": sha256_file(pdf),
                "minimum_ess_fraction": min(data["ess_fractions"]),
                "maximum_calibration_residual": max(data["projection_residuals"]),
                "maximum_absolute_multiplier": max(
                    float(abs(value).max()) for value in data["multipliers"]
                ),
            }
        )
        print(f"saved {png}")
        print(f"saved {pdf}")

    manifest_path = output_dir / "static_visualization_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE_V2_1_CONFIRMED_PARTIAL_STATIC_VISUALIZATIONS",
        "data_role": "INDEPENDENT_HOLDOUT_VISUALIZATION_ONLY",
        "completed_allowance_percentages": [row[0] for row in ALLOWANCES],
        "trial": args.trial,
        "namespace": 23,
        "reference_seed": 310000101,
        "renderer": str(Path(__file__).resolve().relative_to(REPO)),
        "renderer_sha256": sha256_file(Path(__file__)),
        "source_renderer": str((LEGACY_DIR / "visualize_paper.py").relative_to(REPO)),
        "inputs": {
            "config_sha256": sha256_file(args.config),
            "truth_bank_sha256": sha256_file(args.truth_bank),
            "reference_bank_sha256": sha256_file(args.reference_bank),
            "validation_bank_sha256": sha256_file(args.validation_bank),
            "bank_receipt_sha256": sha256_file(BANK_RECEIPT),
            "pause_receipt_sha256": sha256_file(PAUSE_RECEIPT),
            "reference_receipt_sha256": sha256_file(REFERENCE_RECEIPT),
            "selection_geometries_sha256": sha256_file(GEOMETRY_RECEIPT),
            "geometry_adapter_sha256": sha256_file(adapter),
        },
        "artifacts": artifacts,
        "selection_state_changed": False,
    }
    atomic_json(manifest_path, manifest)
    print(f"saved {manifest_path}")
    return manifest


def main() -> int:
    render_static_set(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
