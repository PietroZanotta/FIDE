"""Gated nested Pareto sweep with an explicit, labeled exploratory override."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import jax

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
jax.config.update("jax_enable_x64", True)

from mfsi.io import write_csv, write_json
from experiments.skyrmions_deep_ritz.experiment import run_experiment

DEFAULT_PERCENTAGES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


def _seed_legacy_cache(source: Path, destination: Path, namespace: str) -> None:
    """Seed a content-addressed sweep cache from the certified 3% artifacts."""

    destination.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return
    for json_path in sorted(source.glob("*.json")):
        params_path = json_path.with_suffix(".npz")
        if not params_path.exists():
            continue
        stem = f"legacy-{namespace}-{json_path.stem}"
        shutil.copy2(json_path, destination / f"{stem}.json")
        shutil.copy2(params_path, destination / f"{stem}.npz")


def _resume_certified_etas(output: Path) -> list[list[float]]:
    """Retain only the best certified discovery from an interrupted sweep."""

    best: tuple[float, list[float]] | None = None
    for path in sorted(output.glob("risk_*pct/authoritative_candidates/*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8")).get("result", {})
        except (OSError, json.JSONDecodeError):
            continue
        eta = result.get("eta")
        if not result.get("valid") or not isinstance(eta, list):
            continue
        action = float(result.get("action", float("inf")))
        if not (action < float("inf")):
            continue
        candidate = (action, [float(value) for value in eta])
        if best is None or candidate[0] < best[0]:
            best = candidate
    return [best[1]] if best is not None else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Certified nested skyrmion Deep Ritz Pareto sweep")
    parser.add_argument(
        "--source-result", type=Path,
        default=SCRIPT_DIR / "outputs" / "run" / "result.json",
    )
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs" / "pareto")
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--percent", nargs="+", type=float, default=list(DEFAULT_PERCENTAGES))
    parser.add_argument(
        "--allow-failed-3pct",
        action="store_true",
        help=(
            "run an exploratory sweep despite a failed 3%% validation gate; "
            "default certified behavior remains fail-closed"
        ),
    )
    args = parser.parse_args()
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    validation_path = args.validation_report or args.source_result.parent / "three_percent_validation.json"
    try:
        validation_gate = json.loads(validation_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        validation_gate = {}
    invalidation_path = args.source_result.parent / "anchor_invalidation.json"
    try:
        invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        invalidation = {}
    gate_failures = []
    if source.get("smoke"):
        gate_failures.append("source_is_smoke")
    if not source.get("milestone_success"):
        gate_failures.append("source_milestone_failed")
    if not source.get("pareto_unlocked"):
        gate_failures.append("source_pareto_locked")
    if not validation_gate.get("passed"):
        gate_failures.append("standalone_validation_failed")
    if not validation_gate.get("pareto_unlocked"):
        gate_failures.append("standalone_pareto_locked")
    if (
        invalidation.get("unresolved") is True
        and invalidation.get("invalidated_config_hash") == source.get("config_hash")
    ):
        gate_failures.append("law_anchor_invalidated")
    if gate_failures and not args.allow_failed_3pct:
        raise SystemExit(
            "Pareto remains locked: run the non-smoke 3% experiment and pass every "
            "projection, ESS, Deep Ritz, validation, and action-reduction gate first; "
            "then run validate_3pct.py to create three_percent_validation.json."
        )
    exploratory = bool(gate_failures)
    if exploratory:
        print(
            "WARNING: exploratory Pareto override active; failed gates: "
            + ", ".join(gate_failures),
            flush=True,
        )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source_full_eta = source["full_3_percent"]["eta"]
    incumbent_eta = source_full_eta
    law_eta = source["law_anchor"]["eta"]
    law_risk = float(source["law_anchor"]["risk"])
    source_allowance = float(source["config"]["search"]["risk_allowance_percent"])
    shared_candidate_cache = output / "_shared_cache" / "authoritative"
    shared_validation_cache = output / "_shared_cache" / "validation"
    resumed_etas = _resume_certified_etas(output)
    if resumed_etas:
        print(
            f"[pareto] retaining {len(resumed_etas)} certified designs from prior partial runs",
            flush=True,
        )
    _seed_legacy_cache(
        args.source_result.parent / "authoritative_candidates",
        shared_candidate_cache,
        "source",
    )
    _seed_legacy_cache(
        args.source_result.parent / "validation_candidates",
        shared_validation_cache,
        "source",
    )
    previous_action: float | None = None
    rows = []
    for allowance in sorted(set(float(value) for value in args.percent)):
        tag = f"risk_{allowance:g}pct".replace(".", "p")
        # Reuse the already completed source allowance verbatim when it remains
        # the nested incumbent. If a tighter allowance found a genuinely better
        # candidate, rerun the source allowance with both designs mandatory and let the audited
        # comparison decide, as required by the specification.
        reuse_source = (
            abs(allowance - source_allowance) <= 1.0e-12
            and (
                previous_action is None
                or float(source["full_3_percent"]["selection_action"])
                <= previous_action + 1.0e-10
            )
        )
        if reuse_source:
            result = source
            result_path = args.source_result.resolve()
        else:
            cfg = deepcopy(source["config"])
            cfg["search"]["risk_allowance_percent"] = allowance
            cfg["search"]["fixed_law_eta"] = law_eta
            cfg["search"]["fixed_law_risk"] = law_risk
            mandatory_etas = []
            mandatory_keys = set()
            for eta in (law_eta, source_full_eta, incumbent_eta, *resumed_etas):
                key = tuple(round(float(value), 12) for value in eta)
                if key not in mandatory_keys:
                    mandatory_keys.add(key)
                    mandatory_etas.append(eta)
            cfg["search"]["mandatory_etas"] = mandatory_etas
            result = run_experiment(
                cfg,
                output / tag,
                smoke=False,
                shared_candidate_cache=shared_candidate_cache,
                shared_validation_cache=shared_validation_cache,
                frozen_artifact_source=args.source_result.parent,
            )
            result_path = output / tag / "result.json"
        selected = result["full_3_percent"]
        if not selected["valid"]:
            raise RuntimeError(f"{allowance:g}% winner failed authoritative certification")
        action = float(selected["selection_action"])
        if previous_action is not None and action > previous_action + 1.0e-10:
            raise RuntimeError("nested incumbent invariant violated; action increased")
        previous_action = action
        incumbent_eta = selected["eta"]
        validation = result["validation"]["full"]
        rows.append({
            "allowance_percent": allowance,
            "eta": incumbent_eta,
            "selection_risk": selected["selection_risk"],
            "extra_risk_percent": selected["extra_risk_percent"],
            "budget_used_fraction": selected["extra_risk_percent"] / allowance if allowance else 0.0,
            "selection_action": action,
            "validation_risk": validation["risk"],
            "validation_action": validation["action"],
            "validation_action_standard_error": validation.get("certificate", {}).get("action_standard_error"),
            "validation_law_action": result["validation"]["law"]["action"],
            "validation_action_reduction_vs_law": result["validation_contrast"]["full_vs_law_action_reduction"],
            "action_reduction_vs_law": selected["action_reduction_vs_law"],
            "minimum_ess_fraction": selected["minimum_ess_fraction"],
            "maximum_weak_residual": selected["certificate"]["maximum_weak_residual"],
            "maximum_energy_residual": selected["certificate"]["maximum_energy_residual"],
            "valid": selected["valid"] and validation["valid"],
            "milestone_success": result["milestone_success"],
            "exploratory": exploratory,
            "result": str(result_path),
        })
    write_json(output / "pareto.json", {
        "source_result": str(args.source_result),
        "source_allowance_percent": source_allowance,
        "frozen_law_eta": law_eta,
        "frozen_law_risk": law_risk,
        "resumed_certified_designs": len(resumed_etas),
        "certified": not exploratory,
        "exploratory_override": exploratory,
        "overridden_gate_failures": gate_failures,
        "rows": rows,
    })
    write_csv(output / "pareto.csv", rows)
    print(f"pareto={output / 'pareto.json'}")


if __name__ == "__main__":
    main()
