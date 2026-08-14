#!/usr/bin/env python3
"""
Stage D.8d: local learned-native refinement of D5-Full versus D5-Law.

Scientific purpose
------------------
D8c showed that, under the SAME endpoint-trained D5 interpolant,

    D5-Law  = (24.324, 68.108) deg
    D5-Full = (19.459, 68.108) deg

on the coarse 37-angle D8 grid.  Their independent-validation law risks were
within 1%, but their full-action contrast was noisy.  D8d asks whether the
coarse angular grid and small 8-trial action-selection bank hid a better
learned-native Full-TC design.

This is a local refinement, NOT a relaxation of the law constraint:

  1. Build a fine local angular grid around the D5-Law / D5-Full basin.
  2. Population-law screen exactly as D8:
         L(eta) <= (1 + tau_L) min_eta L(eta).
  3. Finite-resource law screen with CRNs:
         R_N(eta) <= (1 + tau_R) min R_N.
  4. Cheap action pre-screen on every law-feasible candidate.
  5. Re-estimate the best action shortlist with many more paired CRN trials.
  6. Select D5-Full-refined by minimum mean full weighted-Poisson action.
  7. Compare it to the local D5-Law optimum on a FRESH, DISJOINT validation bank.

Crucially, D8d uses a new random-number namespace (default 8890 for selection,
8891 for validation).  It does not tune on the already-inspected D8/D8c
validation trials.

All law, calibration, and action calculations use the same endpoint-trained D5
particle reference and D7 finite-measurement machinery.  No analytic
interpolant is used.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def load_module(path: Path, name: str):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jsonify(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return {k: jsonify(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, Mapping):
        return {str(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, jax.Array):
        a = np.asarray(x)
        return a.item() if a.ndim == 0 else a.tolist()
    return x


def write_json(path: Path, obj: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonify(obj), indent=2, sort_keys=True, allow_nan=True) + "\n")
    tmp.replace(path)


def mean_se(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    se = float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan")
    return {"mean": float(np.mean(x)), "se": se, "n": int(x.size)}


def canonical_eta_deg(pair: Sequence[float]) -> np.ndarray:
    x = np.mod(np.asarray(pair, dtype=np.float64), 180.0)
    x = np.sort(x)
    return x


def projective_sep_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def eta_key(eta_rad: np.ndarray) -> str:
    d = np.degrees(np.mod(np.asarray(eta_rad, dtype=np.float64), np.pi))
    d = np.sort(d)
    return f"{d[0]:.9f},{d[1]:.9f}"


def paired_reduction(rows_num: List[Mapping[str, Any]], rows_den: List[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    """num is the candidate expected to be cheaper; den is comparator.

    ratio_of_means_reduction = 1 - mean(num) / mean(den)
    mean_paired_reduction    = mean_i[1 - num_i / den_i]
    """
    x = np.asarray([r.get(key, np.nan) for r in rows_num], dtype=np.float64)
    y = np.asarray([r.get(key, np.nan) for r in rows_den], dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(y) > 1e-14)
    x, y = x[m], y[m]
    if x.size == 0:
        return {k: float("nan") for k in (
            "ratio_of_means_reduction", "mean_paired_reduction", "se_paired_reduction",
            "median_paired_reduction", "candidate_wins_fraction") } | {"n": 0}
    p = 1.0 - x / y
    return {
        "ratio_of_means_reduction": float(1.0 - np.mean(x) / np.mean(y)),
        "mean_paired_reduction": float(np.mean(p)),
        "se_paired_reduction": float(np.std(p, ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan"),
        "median_paired_reduction": float(np.median(p)),
        "candidate_wins_fraction": float(np.mean(x < y)),
        "mean_candidate": float(np.mean(x)),
        "mean_comparator": float(np.mean(y)),
        "n": int(x.size),
    }


def paired_difference(rows_a: List[Mapping[str, Any]], rows_b: List[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    x = np.asarray([r.get(key, np.nan) for r in rows_a], dtype=np.float64)
    y = np.asarray([r.get(key, np.nan) for r in rows_b], dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    d = x[m] - y[m]
    s = mean_se(d)
    return {"mean_a_minus_b": s["mean"], "se": s["se"], "n": s["n"]}


def bootstrap_reductions(rows_num, rows_den, key: str, seed: int = 12345, reps: int = 5000) -> Dict[str, Any]:
    x = np.asarray([r.get(key, np.nan) for r in rows_num], dtype=np.float64)
    y = np.asarray([r.get(key, np.nan) for r in rows_den], dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(y) > 1e-14)
    x, y = x[m], y[m]
    n = x.size
    if n < 2:
        return {"n": int(n), "reps": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(reps), n))
    xb = x[idx]
    yb = y[idx]
    rom = 1.0 - np.mean(xb, axis=1) / np.mean(yb, axis=1)
    mpr = np.mean(1.0 - xb / yb, axis=1)
    return {
        "n": int(n), "reps": int(reps),
        "ratio_of_means_ci95": np.quantile(rom, [0.025, 0.975]).tolist(),
        "mean_paired_ci95": np.quantile(mpr, [0.025, 0.975]).tolist(),
    }


def write_csv(path: Path, rows: List[Mapping[str, Any]]):
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k, v in r.items():
            if k not in seen and np.isscalar(v):
                seen.add(k); keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# -----------------------------------------------------------------------------
# Local design grid and CRNs
# -----------------------------------------------------------------------------


def local_candidates(args) -> List[Dict[str, Any]]:
    def axis(lo, hi, step):
        n = int(round((hi - lo) / step))
        vals = lo + step * np.arange(n + 1, dtype=np.float64)
        vals = vals[vals <= hi + 1e-10]
        return vals

    a1 = axis(args.theta1_min, args.theta1_max, args.step_deg)
    a2 = axis(args.theta2_min, args.theta2_max, args.step_deg)
    raw = []
    for x in a1:
        for y in a2:
            d = canonical_eta_deg([x, y])
            if projective_sep_deg(d[0], d[1]) + 1e-10 < args.min_sep_deg:
                continue
            raw.append((d, "local_grid"))

    if not args.no_anchors:
        raw += [
            (canonical_eta_deg([24.324324324324323, 68.10810810810811]), "d8c_law_anchor"),
            (canonical_eta_deg([19.45945945945946, 68.10810810810811]), "d8c_full_anchor"),
            (canonical_eta_deg([24.324324324324323, 63.24324324324324]), "d8c_tangent_anchor"),
        ]

    out, seen = [], {}
    for d, source in raw:
        eta = np.radians(d)
        k = eta_key(eta)
        if k in seen:
            out[seen[k]]["sources"].append(source)
        else:
            seen[k] = len(out)
            out.append({"eta": eta, "theta_deg": d.tolist(), "sources": [source]})
    return out


def make_shared_bank(model, c2, cfg, acq_idx, count: int, namespace: int):
    bank = []
    for trial in range(int(count)):
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), int(namespace), int(trial)]))
        alpha = float(rng.uniform(model.cfg.alpha_min, model.cfg.alpha_max))
        bank.append(c2.draw_shared_trial(model, alpha, acq_idx, int(cfg.finite_n), rng))
    return bank


# -----------------------------------------------------------------------------
# Progress cache
# -----------------------------------------------------------------------------


def progress_signature(args, cfg, checkpoint: Path) -> Dict[str, Any]:
    return {
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "preset": args.preset,
        "theta1": [args.theta1_min, args.theta1_max],
        "theta2": [args.theta2_min, args.theta2_max],
        "step_deg": args.step_deg,
        "min_sep_deg": args.min_sep_deg,
        "law_trials": args.law_trials,
        "pre_action_trials": args.pre_action_trials,
        "action_trials": args.action_trials,
        "validation_trials": args.validation_trials,
        "action_shortlist": args.action_shortlist,
        "tau_l": float(cfg.tau_l), "tau_r": float(cfg.tau_r),
        "selection_namespace": args.selection_namespace,
        "validation_namespace": args.validation_namespace,
        "seed": int(cfg.seed),
    }


def load_progress(path: Path, signature: Mapping[str, Any], resume: bool) -> Dict[str, Any]:
    blank = {"signature": jsonify(signature), "population": {}, "finite_law": {}, "pre_action": {}, "refined_action": {}, "validation": {}}
    if not resume or not path.exists():
        return blank
    try:
        p = json.loads(path.read_text())
        if p.get("signature") != jsonify(signature):
            print("NOTE: existing D8d progress signature differs; ignoring it.", flush=True)
            return blank
        for k in blank:
            if k not in p:
                p[k] = blank[k]
        return p
    except Exception as exc:
        print(f"NOTE: could not load D8d progress ({exc}); starting fresh.", flush=True)
        return blank


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d8-script", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--c2-script", required=True)
    p.add_argument("--d2-script", required=True)
    p.add_argument("--d3-script", required=True)
    p.add_argument("--d5-script", required=True)
    p.add_argument("--d7-script", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="reference")

    p.add_argument("--theta1-min", type=float, default=15.0)
    p.add_argument("--theta1-max", type=float, default=27.0)
    p.add_argument("--theta2-min", type=float, default=62.0)
    p.add_argument("--theta2-max", type=float, default=74.0)
    p.add_argument("--step-deg", type=float, default=1.0)
    p.add_argument("--min-sep-deg", type=float, default=20.0)
    p.add_argument("--no-anchors", action="store_true")

    p.add_argument("--law-trials", type=int, default=32,
                   help="CRN trials for finite-law selection.")
    p.add_argument("--pre-action-trials", type=int, default=8,
                   help="Cheap full-action pre-screen over every law-feasible candidate.")
    p.add_argument("--action-trials", type=int, default=32,
                   help="Total CRN action trials for shortlisted candidates (includes pre-action trials).")
    p.add_argument("--action-shortlist", type=int, default=12,
                   help="Number of best pre-screen candidates advanced to the larger action bank.")
    p.add_argument("--validation-trials", type=int, default=64,
                   help="Fresh disjoint validation trials for local D5-Law and refined D5-Full.")
    p.add_argument("--selection-namespace", type=int, default=8890)
    p.add_argument("--validation-namespace", type=int, default=8891)
    p.add_argument("--bootstrap-reps", type=int, default=5000)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--output", default="stage_d8d_local_refined_full_vs_law.json")
    return p


def main():
    t0 = time.time()
    args = parser().parse_args()
    if args.step_deg <= 0:
        raise ValueError("--step-deg must be >0")
    if args.pre_action_trials < 1 or args.action_trials < args.pre_action_trials:
        raise ValueError("Need 1 <= pre-action-trials <= action-trials")
    if args.law_trials < 2 or args.validation_trials < 2:
        raise ValueError("law-trials and validation-trials must be >=2")
    if args.action_shortlist < 1:
        raise ValueError("action-shortlist must be >=1")
    if args.selection_namespace == args.validation_namespace:
        raise ValueError("Selection and validation CRN namespaces must differ")

    d8 = load_module(Path(args.d8_script), "stage_d8_for_d8d")
    backend = load_module(Path(args.backend), "stage_b_for_d8d")
    c2 = load_module(Path(args.c2_script), "stage_c2_for_d8d")
    d2 = load_module(Path(args.d2_script), "stage_d2_for_d8d")
    d3 = load_module(Path(args.d3_script), "stage_d3_for_d8d")
    d5 = load_module(Path(args.d5_script), "stage_d5_for_d8d")
    d7 = load_module(Path(args.d7_script), "stage_d7_for_d8d")
    if hasattr(d8, "install_robust_d3_projection"):
        d8.install_robust_d3_projection(d3)

    params, checkpoint_meta = d5.load_checkpoint(Path(args.checkpoint))
    cfg0 = d8.preset_d8_config(args.preset)
    # Keep physical/numerical preset unchanged; only replace trial counts relevant here.
    cfg = dataclasses.replace(
        cfg0,
        law_trials=int(args.law_trials),
        action_trials=int(args.action_trials),
        validation_trials=int(args.validation_trials),
    )

    base_b = backend.preset_config("quick" if args.preset == "quick" else "reference")
    stage_b_cfg = dataclasses.replace(base_b, grid_n=int(cfg.grid_n), time_n=int(cfg.time_n))
    model = backend.StageB(stage_b_cfg)
    d7_cfg = d8.d7_compatible_config(cfg, d7)
    evaluator = d7.D7Evaluator(model, d2, d5, params, checkpoint_meta, d7_cfg)
    measurement_cov = d3.MeasurementCovariance(model)

    acq_sets = c2.nested_acquisition_sets(model.cfg.time_n, [int(cfg.acquisition_k)])
    acq_idx = np.asarray(acq_sets[int(cfg.acquisition_k)], dtype=int)
    acq_set = set(acq_idx.tolist())
    heldout_idx = np.asarray([i for i in range(model.cfg.time_n) if i not in acq_set and i not in (0, model.cfg.time_n - 1)], dtype=int)
    heldout_mask = np.zeros(model.cfg.time_n, dtype=bool)
    heldout_mask[heldout_idx] = True

    candidates = local_candidates(args)
    max_sel = max(int(args.law_trials), int(args.action_trials))
    selection_bank = make_shared_bank(model, c2, cfg, acq_idx, max_sel, int(args.selection_namespace))
    law_bank = selection_bank[: int(args.law_trials)]
    action_bank = selection_bank[: int(args.action_trials)]
    validation_bank = make_shared_bank(model, c2, cfg, acq_idx, int(args.validation_trials), int(args.validation_namespace))

    output = Path(args.output).expanduser().resolve()
    progress_path = output.with_suffix(".progress.json")
    sig = progress_signature(args, cfg, Path(args.checkpoint))
    progress = load_progress(progress_path, sig, resume=not args.no_resume)

    print("=" * 120)
    print("Stage D.8d — local learned-native refinement: D5-Full vs D5-Law")
    print("=" * 120)
    print(f"D5 checkpoint : {Path(args.checkpoint).expanduser().resolve()}")
    print(f"Grid/time     : {model.cfg.grid_n}x{model.cfg.grid_n} / {model.cfg.time_n}")
    print(f"D5 bank       : {cfg.bank_mode}, GH order={cfg.gh_order}, Nbank={evaluator.x0.shape[0]}")
    print(f"Local grid    : theta1=[{args.theta1_min:g},{args.theta1_max:g}], theta2=[{args.theta2_min:g},{args.theta2_max:g}], step={args.step_deg:g} deg")
    print(f"Candidates    : {len(candidates)} (including exact D8c anchors unless disabled)")
    print(f"Law tolerance : tau_L={cfg.tau_l:.3f}, tau_R={cfg.tau_r:.3f}")
    print(f"Trials        : law={args.law_trials}, pre-action={args.pre_action_trials}, refined-action={args.action_trials}, fresh-validation={args.validation_trials}")
    print(f"CRN namespaces: selection={args.selection_namespace}, validation={args.validation_namespace}")

    # ------------------------------------------------------------------
    # A. Population-law screen.
    # ------------------------------------------------------------------
    print("\n[A] Population-law screen on fine local grid...", flush=True)
    for j, c in enumerate(candidates):
        k = eta_key(c["eta"])
        if k in progress["population"]:
            c["population"] = progress["population"][k]
        else:
            c["population"] = d8.population_endpoint_fm_law(model, evaluator, d2, d7, np.asarray(c["eta"]), cfg)
            progress["population"][k] = jsonify(c["population"])
            write_json(progress_path, progress)
        if (j + 1) % max(1, len(candidates) // 10) == 0 or j + 1 == len(candidates):
            print(f"  population {j+1}/{len(candidates)}", flush=True)

    pop_valid = [c for c in candidates if bool(c["population"].get("scientifically_valid", False)) and np.isfinite(c["population"].get("lift_mmd2", np.nan))]
    if not pop_valid:
        raise RuntimeError("No numerically valid local population candidates")
    lstar = min(float(c["population"]["lift_mmd2"]) for c in pop_valid)
    lmax = (1.0 + float(cfg.tau_l)) * lstar
    pop_feasible = [c for c in pop_valid if float(c["population"]["lift_mmd2"]) <= lmax + 1e-15]
    print(f"  L*={lstar:.8e}; threshold={lmax:.8e}; population-feasible={len(pop_feasible)}/{len(pop_valid)}", flush=True)

    # Common feasibility geometry only where needed for finite measurement work.
    print("  Building common physical/D5-particle feasibility polytopes...", flush=True)
    feasible_for_finite = []
    for j, c in enumerate(pop_feasible):
        try:
            c["constraints"] = d7.build_joint_beta_constraints(
                model=model, evaluator=evaluator, d2=d2, c2=c2,
                eta=np.asarray(c["eta"]), margin=float(cfg.feasibility_margin),
            )
            feasible_for_finite.append(c)
        except Exception as exc:
            c["constraint_error"] = repr(exc)
        if (j + 1) % max(1, len(pop_feasible) // 10) == 0 or j + 1 == len(pop_feasible):
            print(f"    constraints {j+1}/{len(pop_feasible)}", flush=True)

    # ------------------------------------------------------------------
    # B. Finite-resource law screen on fresh selection CRNs.
    # ------------------------------------------------------------------
    print("\n[B] Finite-resource law screen on fresh CRNs...", flush=True)
    for j, c in enumerate(feasible_for_finite):
        k = eta_key(c["eta"])
        cached = progress["finite_law"].get(k)
        if cached is not None:
            c["finite_law"] = cached
        else:
            vals, exact_vals = [], []
            valid = True
            for shared in law_bank:
                row, _ = d7.evaluate_design_trial(
                    model=model, evaluator=evaluator, d2=d2, d3=d3, c2=c2,
                    measurement_cov=measurement_cov, eta=np.asarray(c["eta"]), shared=shared,
                    acq_idx=acq_idx, heldout_mask=heldout_mask, cfg=d7_cfg,
                    joint_constraints=c["constraints"], compute_action=False,
                )
                vals.append(row.get("finite_heldout_mmd2", np.nan))
                exact_vals.append(row.get("exact_heldout_mmd2", np.nan))
                valid = valid and bool(row.get("finite_valid", False)) and bool(row.get("exact_valid", False))
            rec = {
                "finite_risk": mean_se(vals),
                "exact_risk": mean_se(exact_vals),
                "valid": bool(valid and np.all(np.isfinite(vals))),
            }
            c["finite_law"] = rec
            progress["finite_law"][k] = jsonify(rec)
            write_json(progress_path, progress)
        print(
            f"  law {j+1}/{len(feasible_for_finite)} eta={np.degrees(c['eta']).round(3).tolist()} | "
            f"R={c['finite_law']['finite_risk']['mean']:.8e} +/- {c['finite_law']['finite_risk']['se']:.2e} | valid={c['finite_law']['valid']}",
            flush=True,
        )

    finite_valid = [c for c in feasible_for_finite if c["finite_law"]["valid"]]
    if not finite_valid:
        raise RuntimeError("No valid finite-law candidates")
    law_best = min(finite_valid, key=lambda c: float(c["finite_law"]["finite_risk"]["mean"]))
    rstar = float(law_best["finite_law"]["finite_risk"]["mean"])
    rmax = (1.0 + float(cfg.tau_r)) * rstar
    law_feasible = [c for c in finite_valid if float(c["finite_law"]["finite_risk"]["mean"]) <= rmax + 1e-15]
    print(f"  -> local D5-Law eta={np.degrees(law_best['eta']).round(6).tolist()} | R*={rstar:.8e}; 1% set={len(law_feasible)}/{len(finite_valid)}", flush=True)

    # ------------------------------------------------------------------
    # C1. Action pre-screen on all law-feasible candidates.
    # ------------------------------------------------------------------
    print("\n[C1] Full-action pre-screen over law-feasible candidates...", flush=True)
    for j, c in enumerate(law_feasible):
        k = eta_key(c["eta"])
        cached = progress["pre_action"].get(k)
        if cached is not None and len(cached.get("rows", [])) >= int(args.pre_action_trials):
            rec = cached
        else:
            rows = [] if cached is None else list(cached.get("rows", []))
            for trial_i in range(len(rows), int(args.pre_action_trials)):
                row, _ = d7.evaluate_design_trial(
                    model=model, evaluator=evaluator, d2=d2, d3=d3, c2=c2,
                    measurement_cov=measurement_cov, eta=np.asarray(c["eta"]), shared=action_bank[trial_i],
                    acq_idx=acq_idx, heldout_mask=heldout_mask, cfg=d7_cfg,
                    joint_constraints=c["constraints"], compute_action=True,
                )
                rows.append(row)
            rec = {
                "rows": rows,
                "action": mean_se([r.get("finite_action", np.nan) for r in rows]),
                "valid": bool(all(bool(r.get("finite_valid", False)) and np.isfinite(r.get("finite_action", np.nan)) for r in rows)),
            }
            progress["pre_action"][k] = jsonify(rec)
            write_json(progress_path, progress)
        c["pre_action"] = rec
        print(f"  pre {j+1}/{len(law_feasible)} eta={np.degrees(c['eta']).round(3).tolist()} | A={rec['action']['mean']:.6e} | valid={rec['valid']}", flush=True)

    pre_valid = [c for c in law_feasible if c["pre_action"]["valid"]]
    if not pre_valid:
        raise RuntimeError("No valid pre-action candidates")
    pre_sorted = sorted(pre_valid, key=lambda c: float(c["pre_action"]["action"]["mean"]))
    shortlist = pre_sorted[: min(int(args.action_shortlist), len(pre_sorted))]

    # Always carry local law-best and exact D8c Full anchor if law-feasible.
    must_keys = {eta_key(law_best["eta"]), eta_key(np.radians([19.45945945945946, 68.10810810810811]))}
    short_map = {eta_key(c["eta"]): c for c in shortlist}
    for c in pre_valid:
        if eta_key(c["eta"]) in must_keys:
            short_map[eta_key(c["eta"])] = c
    shortlist = list(short_map.values())
    shortlist.sort(key=lambda c: float(c["pre_action"]["action"]["mean"]))
    print(f"  -> advancing {len(shortlist)} candidates to {args.action_trials}-trial refined action bank", flush=True)

    # ------------------------------------------------------------------
    # C2. Refined action estimates.
    # ------------------------------------------------------------------
    print("\n[C2] Refined full-action estimation on shortlist...", flush=True)
    for j, c in enumerate(shortlist):
        k = eta_key(c["eta"])
        cached = progress["refined_action"].get(k)
        if cached is not None and len(cached.get("rows", [])) >= int(args.action_trials):
            rec = cached
        else:
            if cached is not None:
                rows = list(cached.get("rows", []))
            else:
                rows = list(c["pre_action"]["rows"])
            for trial_i in range(len(rows), int(args.action_trials)):
                row, _ = d7.evaluate_design_trial(
                    model=model, evaluator=evaluator, d2=d2, d3=d3, c2=c2,
                    measurement_cov=measurement_cov, eta=np.asarray(c["eta"]), shared=action_bank[trial_i],
                    acq_idx=acq_idx, heldout_mask=heldout_mask, cfg=d7_cfg,
                    joint_constraints=c["constraints"], compute_action=True,
                )
                rows.append(row)
            rec = {
                "rows": rows,
                "action": mean_se([r.get("finite_action", np.nan) for r in rows]),
                "valid": bool(all(bool(r.get("finite_valid", False)) and np.isfinite(r.get("finite_action", np.nan)) for r in rows)),
            }
            progress["refined_action"][k] = jsonify(rec)
            write_json(progress_path, progress)
        c["refined_action"] = rec
        print(f"  refined {j+1}/{len(shortlist)} eta={np.degrees(c['eta']).round(3).tolist()} | A={rec['action']['mean']:.6e} +/- {rec['action']['se']:.2e} | valid={rec['valid']}", flush=True)

    refined_valid = [c for c in shortlist if c["refined_action"]["valid"]]
    if not refined_valid:
        raise RuntimeError("No valid refined-action candidates")
    full_best = min(refined_valid, key=lambda c: float(c["refined_action"]["action"]["mean"]))
    print("\nSelected local learned-native designs:")
    print(f"  D5-Law refined : {np.degrees(law_best['eta']).round(6).tolist()} | R={rstar:.8e}")
    print(f"  D5-Full refined: {np.degrees(full_best['eta']).round(6).tolist()} | A_sel={full_best['refined_action']['action']['mean']:.6e}")

    # ------------------------------------------------------------------
    # D. Fresh independent validation.  Evaluate both designs on every trial.
    # ------------------------------------------------------------------
    print("\n[D] Fresh independent validation of refined D5-Law vs D5-Full...", flush=True)
    selected = {"d5_law": law_best, "d5_full_refined": full_best}
    validation_rows: Dict[str, List[Dict[str, Any]]] = {}
    for name, c in selected.items():
        k = eta_key(c["eta"])
        cached = progress["validation"].get(k)
        rows = [] if cached is None else list(cached.get("rows", []))
        for trial_i in range(len(rows), int(args.validation_trials)):
            row, _ = d7.evaluate_design_trial(
                model=model, evaluator=evaluator, d2=d2, d3=d3, c2=c2,
                measurement_cov=measurement_cov, eta=np.asarray(c["eta"]), shared=validation_bank[trial_i],
                acq_idx=acq_idx, heldout_mask=heldout_mask, cfg=d7_cfg,
                joint_constraints=c["constraints"], compute_action=True,
            )
            rr = dict(row)
            rr["validation_trial"] = int(trial_i)
            rows.append(rr)
            progress["validation"][k] = {"rows": jsonify(rows)}
            write_json(progress_path, progress)
        validation_rows[name] = rows
        rs = mean_se([r.get("finite_heldout_mmd2", np.nan) for r in rows])
        aa = mean_se([r.get("finite_action", np.nan) for r in rows])
        print(f"  {name:15s} eta={np.degrees(c['eta']).round(3).tolist()} | R={rs['mean']:.8e} +/- {rs['se']:.2e} | A={aa['mean']:.6e} +/- {aa['se']:.2e}", flush=True)

    law_rows = validation_rows["d5_law"]
    full_rows = validation_rows["d5_full_refined"]
    reduction = paired_reduction(full_rows, law_rows, "finite_action")
    reduction_boot = bootstrap_reductions(full_rows, law_rows, "finite_action", seed=int(cfg.seed) + 991, reps=int(args.bootstrap_reps))
    law_diff = paired_difference(full_rows, law_rows, "finite_heldout_mmd2")
    law_mean = mean_se([r.get("finite_heldout_mmd2", np.nan) for r in law_rows])
    full_mean = mean_se([r.get("finite_heldout_mmd2", np.nan) for r in full_rows])
    full_law_relative = float(full_mean["mean"] / law_mean["mean"] - 1.0)

    print("\n" + "=" * 120)
    print("D8d fresh-validation result")
    print("=" * 120)
    print(f"D5-Law         eta={np.degrees(law_best['eta']).round(6).tolist()} | R={law_mean['mean']:.8e}")
    print(f"D5-Full-refined eta={np.degrees(full_best['eta']).round(6).tolist()} | R={full_mean['mean']:.8e} | law penalty={100*full_law_relative:+.3f}%")
    print(
        f"Full-vs-Law action reduction: ratio-of-means={100*reduction['ratio_of_means_reduction']:+.2f}% | "
        f"mean-paired={100*reduction['mean_paired_reduction']:+.2f}% +/- {100*reduction['se_paired_reduction']:.2f}% | "
        f"median-paired={100*reduction['median_paired_reduction']:+.2f}% | wins={100*reduction['candidate_wins_fraction']:.1f}%"
    )
    if reduction_boot.get("reps", 0):
        romci = 100*np.asarray(reduction_boot["ratio_of_means_ci95"])
        mpci = 100*np.asarray(reduction_boot["mean_paired_ci95"])
        print(f"Bootstrap 95% CI: ratio-of-means=[{romci[0]:+.2f}%, {romci[1]:+.2f}%] | mean-paired=[{mpci[0]:+.2f}%, {mpci[1]:+.2f}%]")

    def public_candidate(c):
        return {
            "theta_deg": np.degrees(c["eta"]).tolist(),
            "sources": c.get("sources", []),
            "population_law": c.get("population", {}).get("lift_mmd2", np.nan),
            "finite_law": c.get("finite_law"),
            "pre_action": {k: v for k, v in c.get("pre_action", {}).items() if k != "rows"},
            "refined_action": {k: v for k, v in c.get("refined_action", {}).items() if k != "rows"},
        }

    payload = {
        "stage": "D.8d local learned-native refinement of D5-Full versus D5-Law",
        "created_unix_time": time.time(),
        "python": platform.python_version(), "jax": jax.__version__,
        "paths": {
            "d8_script": str(Path(args.d8_script).resolve()),
            "backend": str(Path(args.backend).resolve()), "c2_script": str(Path(args.c2_script).resolve()),
            "d2_script": str(Path(args.d2_script).resolve()), "d3_script": str(Path(args.d3_script).resolve()),
            "d5_script": str(Path(args.d5_script).resolve()), "d7_script": str(Path(args.d7_script).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
        },
        "config": jsonify(cfg), "stage_b_config": jsonify(stage_b_cfg),
        "local_search": {
            "theta1_deg": [args.theta1_min, args.theta1_max], "theta2_deg": [args.theta2_min, args.theta2_max],
            "step_deg": args.step_deg, "min_sep_deg": args.min_sep_deg,
            "candidate_count": len(candidates), "population_feasible_count": len(pop_feasible),
            "finite_law_feasible_count": len(law_feasible), "action_shortlist_count": len(shortlist),
            "L_star": lstar, "L_max": lmax, "R_star": rstar, "R_max": rmax,
        },
        "randomness": {
            "seed": int(cfg.seed), "selection_namespace": int(args.selection_namespace),
            "validation_namespace": int(args.validation_namespace),
            "law_trials": int(args.law_trials), "pre_action_trials": int(args.pre_action_trials),
            "action_trials": int(args.action_trials), "validation_trials": int(args.validation_trials),
            "note": "Selection and validation use disjoint fresh CRN namespaces; neither reuses inspected D8/D8c validation trials.",
        },
        "selected": {"d5_law": public_candidate(law_best), "d5_full_refined": public_candidate(full_best)},
        "candidate_summary": [public_candidate(c) for c in candidates],
        "validation": {
            "d5_law": {
                "finite_law": law_mean,
                "finite_action": mean_se([r.get("finite_action", np.nan) for r in law_rows]),
            },
            "d5_full_refined": {
                "finite_law": full_mean,
                "finite_action": mean_se([r.get("finite_action", np.nan) for r in full_rows]),
            },
            "full_relative_law_penalty": full_law_relative,
            "full_vs_law_law_difference": law_diff,
            "full_vs_law_action_reduction": reduction,
            "full_vs_law_action_bootstrap": reduction_boot,
            "trial_rows": validation_rows,
        },
        "interpretation_notes": [
            "The finite-law tolerance remains fixed at D8 tau_R; D8d does not loosen law equivalence to manufacture an action gain.",
            "The refined Full design is selected only from candidates that pass both population and finite-resource law screens.",
            "Action pre-screening is used only for compute efficiency; the final shortlist is re-estimated on the larger action bank and then tested on a fresh validation bank.",
            "Ratio-of-means and mean paired percentage reductions answer different estimands and are both reported, together with bootstrap intervals and the fraction of paired trials won.",
            "No analytic interpolant is used.",
        ],
        "wall_seconds": float(time.time() - t0),
    }
    write_json(output, payload)

    candidate_csv = output.with_suffix(".candidate_summary.csv")
    rows = []
    for c in candidates:
        rows.append({
            "theta1_deg": float(np.degrees(c["eta"])[0]), "theta2_deg": float(np.degrees(c["eta"])[1]),
            "population_law": c.get("population", {}).get("lift_mmd2", np.nan),
            "finite_risk": c.get("finite_law", {}).get("finite_risk", {}).get("mean", np.nan),
            "finite_risk_se": c.get("finite_law", {}).get("finite_risk", {}).get("se", np.nan),
            "pre_action": c.get("pre_action", {}).get("action", {}).get("mean", np.nan),
            "refined_action": c.get("refined_action", {}).get("action", {}).get("mean", np.nan),
            "is_law_best": int(eta_key(c["eta"]) == eta_key(law_best["eta"])),
            "is_full_best": int(eta_key(c["eta"]) == eta_key(full_best["eta"])),
        })
    write_csv(candidate_csv, rows)

    val_csv = output.with_suffix(".validation_trials.csv")
    flat = []
    for name, rws in validation_rows.items():
        for r in rws:
            rr = {"design": name, "theta1_deg": float(np.degrees(selected[name]["eta"])[0]), "theta2_deg": float(np.degrees(selected[name]["eta"])[1])}
            rr.update(r)
            flat.append(rr)
    write_csv(val_csv, flat)

    print(f"Saved JSON     : {output}")
    print(f"Saved progress : {progress_path}")
    print(f"Saved CSV      : {candidate_csv}")
    print(f"Saved CSV      : {val_csv}")


if __name__ == "__main__":
    main()
