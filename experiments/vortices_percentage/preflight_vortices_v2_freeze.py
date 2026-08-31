#!/usr/bin/env python3
"""Fail-closed, outcome-free preflight for the Vortices V2 prospective freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from selection_contract import (
    EVALUATOR_IDENTITY,
    load_selection_config,
    sha256_file,
    validate_selection_config,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_MANIFEST = HERE / "VORTICES_V2_FREEZE_MANIFEST.json"
TEXT_SUFFIXES = {".py", ".json", ".md", ".csv", ".toml", ".yaml", ".yml"}


class PreflightFailure(RuntimeError):
    pass


def _pass(checks: list[dict[str, Any]], name: str, detail: Any) -> None:
    checks.append({"check": name, "status": "PASS", "detail": detail})


def _require(condition: bool, name: str, detail: Any) -> None:
    if not condition:
        raise PreflightFailure(f"{name}: {detail}")


def _check_hash_map(
    checks: list[dict[str, Any]], mapping: dict[str, str], label: str
) -> None:
    for relative, expected in mapping.items():
        path = REPO_ROOT / relative
        _require(path.is_file(), f"{label} exists", relative)
        actual = sha256_file(path)
        _require(actual == expected, f"{label} hash", f"{relative}: {actual} != {expected}")
    _pass(checks, f"{label} hashes", len(mapping))


def _v2_inventory() -> set[str]:
    paths: set[str] = set()
    for path in HERE.rglob("*"):
        if not path.is_file() or "outputs" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix in {".py", ".json", ".md"}:
            paths.add(str(path.relative_to(REPO_ROOT)))
    return paths


def _fresh_values_absent_from_history(values: list[int]) -> tuple[bool, str]:
    pattern = "(^|[^0-9])(" + "|".join(map(str, values)) + ")([^0-9]|$)"
    paths = [
        "experiments/vortices_percentage",
        "experiments/vortices_prospective",
        "testing/FIDE/experiments/vortices_percentage",
        "testing/FIDE/experiments/vortices_prospective",
    ]
    command = ["git", "log", "--all", "--format=%H", f"-G{pattern}", "--", *paths]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    commits = [line for line in result.stdout.splitlines() if line.strip()]
    return not commits, ",".join(commits[:5])


def _fresh_values_absent_from_historical_tree(values: list[int]) -> tuple[bool, list[str]]:
    pattern = re.compile(r"(?<![0-9])(" + "|".join(map(str, values)) + r")(?![0-9])")
    roots = [
        REPO_ROOT / "experiments/vortices_percentage",
        REPO_ROOT / "experiments/vortices_prospective",
        REPO_ROOT / "testing/FIDE/experiments/vortices_percentage",
        REPO_ROOT / "testing/FIDE/experiments/vortices_prospective",
    ]
    matches: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if pattern.search(text):
                matches.append(str(path.relative_to(REPO_ROOT)))
    return not matches, matches[:20]


def run_preflight(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    check_git_history: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    _require(
        manifest.get("status") == "FROZEN_PROSPECTIVE_NO_OUTCOMES",
        "manifest status",
        manifest.get("status"),
    )

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    _require(head == manifest["repository"]["head"], "repository HEAD", head)
    _pass(checks, "repository HEAD", head)

    _check_hash_map(checks, manifest["frozen_files"], "frozen scientific file")
    _check_hash_map(checks, manifest["shared_dependencies"], "shared dependency")
    _check_hash_map(checks, manifest["toy_immutability"], "Toy immutable")
    _check_hash_map(checks, manifest["v1_immutability"], "V1 immutable")

    expected_inventory = set(manifest["frozen_files"]) | {
        str(manifest_path.relative_to(REPO_ROOT))
    }
    actual_inventory = _v2_inventory()
    _require(
        actual_inventory == expected_inventory,
        "recorded V2 scientific inventory",
        {
            "unrecorded": sorted(actual_inventory - expected_inventory),
            "missing": sorted(expected_inventory - actual_inventory),
        },
    )
    _pass(checks, "recorded V2 scientific inventory", len(actual_inventory))

    selection_path = REPO_ROOT / manifest["paths"]["selection_config"]
    config = load_selection_config(selection_path)
    validate_selection_config(config)
    _pass(checks, "optimizer/search config completeness", True)

    numerical = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    evaluator = config["scientific_evaluator"]
    _require(evaluator["identity"] == EVALUATOR_IDENTITY, "reflected evaluator identity", evaluator)
    _require(float(evaluator["density_floor"]) == 0.0, "zero density floor", evaluator)
    _require(evaluator["exact_grid"] == [256, 128], "exact grid", evaluator["exact_grid"])
    _require(evaluator["scientific_time_indices"] == list(range(21)), "21 time nodes", None)
    _require(evaluator["precision"] == "float64", "float64", evaluator["precision"])
    _require(numerical["raster"]["source_column_normalization"] is False, "no column normalization", None)
    _require(numerical["raster"]["reflected_image_pairs"] == 4, "four image pairs", None)
    _pass(checks, "reflection/Neumann evaluator", EVALUATOR_IDENTITY)

    selection_source = (HERE / "selection_contract.py").read_text(encoding="utf-8")
    core_source = (HERE / "core.py").read_text(encoding="utf-8")
    forbidden_calls = (
        "rasterize_projected_particles_positive_rect",
        "rasterize_projected_particles_rect(",
        "0.35 *",
    )
    _require(
        not any(token in selection_source or token in core_source for token in forbidden_calls),
        "legacy raster unreachable from V2 selection",
        forbidden_calls,
    )
    _require("rasterize_trajectory_v2" in selection_source, "selection calls V2 raster", None)
    _pass(checks, "legacy raster unreachable from V2 selection", True)

    numerical_authority_record = manifest["authorities"]["numerical_prefreeze"]
    authority_path = REPO_ROOT / numerical_authority_record["path"]
    _require(
        sha256_file(authority_path) == numerical_authority_record["sha256"],
        "numerical prefreeze authority hash",
        str(authority_path.relative_to(REPO_ROOT)),
    )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    decision = authority["decision_gate"]
    required_authority = (
        "grid_convergence_demonstrated", "grid_action_changes_shrink",
        "particle_convergence_demonstrated", "independent_solver_agreement",
        "manufactured_reflected_continuity", "golden_reflected_continuity",
        "continuity_epsilon_monotone", "common_raster_decomposition",
        "ready_to_freeze_before_reference_training_and_new_selection",
    )
    _require(all(decision[key] for key in required_authority), "numerical prefreeze authority", decision)
    _require(not decision["soft_fiber_required"], "soft fiber remains disabled", decision)
    _pass(checks, "numerical prefreeze authority", authority["status"])

    proxy_authority_record = manifest["authorities"]["search_proxy"]
    proxy_path = REPO_ROOT / proxy_authority_record["path"]
    _require(
        sha256_file(proxy_path) == proxy_authority_record["sha256"],
        "search proxy authority hash",
        str(proxy_path.relative_to(REPO_ROOT)),
    )
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    _require(proxy["status"] == "PASS", "search proxy diagnostic", proxy.get("status"))
    _require(proxy["proxy_grid"] == [64, 32] and proxy["exact_grid"] == [256, 128], "proxy grids", proxy)
    _require(proxy["time_indices"] == list(range(21)), "proxy time nodes", None)
    _require(proxy["selection_config_sha256"] == sha256_file(selection_path), "proxy config hash", None)
    _pass(checks, "V2 search proxy", {"pearson": proxy["overall_pearson"], "spearman": proxy["overall_spearman"]})

    seeds = list(map(int, config["reference_replicates"]["training_seeds"]))
    _require(seeds == manifest["prospective"]["reference_training_seeds"], "frozen seed list", seeds)
    absent, matches = _fresh_values_absent_from_historical_tree(seeds)
    _require(absent, "fresh seeds absent from historical current tree", matches)
    if check_git_history:
        absent_history, commits = _fresh_values_absent_from_history(seeds)
        _require(absent_history, "fresh seeds absent from relevant Git history", commits)
    _pass(checks, "three genuinely fresh reference seeds", seeds)

    banks = config["observation_banks"]
    _require(banks["selection_namespace"] == manifest["prospective"]["selection_namespace"], "selection namespace", None)
    _require(banks["validation_namespace"] == manifest["prospective"]["validation_namespace"], "validation namespace", None)
    _require(banks["selection_namespace"] != banks["validation_namespace"], "namespace disjointness", None)
    _require(banks["shared_across_all_references_and_methods"], "shared observations", None)
    _pass(checks, "shared and disjoint observation namespaces", [banks["selection_namespace"], banks["validation_namespace"]])

    common = config["common_bandwidth"]
    _require(common["ensemble_rule"] == "median_of_exactly_three_qualified_reference_bandwidths", "bandwidth rule", common)
    _require(common["performance_tuning_forbidden"], "bandwidth performance prohibition", None)
    _pass(checks, "median-of-three reference-only bandwidth", True)

    validation = config["validation"]
    _require(validation["bootstrap_resamples"] == 100000 and validation["bootstrap_seed"] == 821775, "bootstrap constants", validation)
    _require("one_common_1024_index_vector" in validation["bootstrap_pairing"], "cross-reference bootstrap pairing", validation["bootstrap_pairing"])
    _pass(checks, "cross-method and cross-reference bootstrap pairing", True)

    prospective_root = REPO_ROOT / config["artifact_destinations"]["root"]
    forbidden_outputs = []
    if prospective_root.exists():
        forbidden_outputs = [
            str(path.relative_to(REPO_ROOT)) for path in prospective_root.rglob("*")
            if path.is_file()
        ]
    _require(not forbidden_outputs, "no prospective scientific outputs", forbidden_outputs[:20])
    _pass(checks, "no new reference/selection/validation files", True)

    return {
        "status": "PASS",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "checks": checks,
        "scientific_operations_performed": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skip-git-history", action="store_true", help="tests only")
    args = parser.parse_args()
    try:
        result = run_preflight(args.manifest, check_git_history=not args.skip_git_history)
    except (PreflightFailure, KeyError, ValueError, FileNotFoundError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
