#!/usr/bin/env python3
"""
Stage D.8b diagnostic: WHY does the endpoint-trained D.8 sensor behave differently
from the historical Full-TC sensor?

This is NOT another sensor oracle. It is a paired mechanism/attribution study.
It holds fixed:
  * the Stage-B external scientific law,
  * the D.5 endpoint-trained FM reference,
  * the D.7 finite-measurement model,
  * the common-feasibility reconstruction,
  * the weighted-Poisson action definition,
  * and the D.8 disjoint validation CRN bank.

Only the sensor geometry eta changes:
    selected D.8 : default (19.459459..., 68.108108...) deg
    historical Full-TC: default (0, 160) deg

The script answers four questions.

1. Is the D.8-vs-Full law gap already present with exact population moments?
   If yes, finite measurement noise is not the explanation.

2. Does D.8 extract more of the scientific-law discrepancy from the raw D.5
   reference?  We compare raw-reference MMD with I-projected MMD and compute
   raw-reference moment mismatches.

3. Does D.8 measure the scientifically varying alpha direction more strongly?
   We compute between-alpha covariance of the target moment vector c_eta(t,alpha).

4. Why can Full have lower action while worse law fidelity?
   We compare ESS/lambda/action and two zero-tilt local proxies:
      static mismatch  (c - E_ref Phi)^T Cov_ref(Phi)^-1 (c - E_ref Phi)
      dynamic mismatch r^T G^-1 r,
   where r = c_dot - E_ref[J Phi u].
   These are diagnostics, NOT replacements for the exact I-projected Poisson action.

Recommended run (reuses the validation JSON you already generated):

python stage_d8b_d8_vs_full_mechanism.py \\
    --d8-script stage_d8_endpoint_flow_matching_finite_resource_design_oracle_fixed.py \\
    --backend ../stage_b/stage_b2_transport_conditioned_design.py \\
    --c2-script ../stage_c/stage_c2_mfsi_matched_action.py \\
    --d2-script stage_d2_flow_matching_particle_mfsi.py \\
    --d3-script stage_d3_flow_matching_finite_measurements.py \\
    --d5-script stage_d5_endpoint_flow_matching_reference_v2.py \\
    --d7-script stage_d7_endpoint_flow_matching_finite_measurements_v2.py \\
    --checkpoint stage_d5_endpoint_flow_matching_reference_v2.npz \\
    --preset reference \\
    --validation-json stage_d8_endpoint_flow_matching_finite_resource_design_oracle_validation_resume.json \\
    --output stage_d8b_d8_vs_full_mechanism.json

If --validation-json is absent or does not contain the required trial rows, the
script recomputes ONLY D.8 and Full on the exact original D.8 validation bank.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


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


def atomic_json(path: Path, payload: Mapping[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonify(payload), indent=2, sort_keys=True, allow_nan=True) + "\n")
    tmp.replace(path)


def weighted_cov(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = w / max(float(np.sum(w)), 1e-300)
    x = np.asarray(x, dtype=np.float64)
    mu = np.sum(w[:, None] * x, axis=0)
    xc = x - mu[None, :]
    return xc.T @ (w[:, None] * xc)


def sym_pinv(a: np.ndarray, ridge: float = 1e-12) -> np.ndarray:
    a = 0.5 * (np.asarray(a, dtype=np.float64) + np.asarray(a, dtype=np.float64).T)
    scale = max(float(np.trace(a)) / max(a.shape[0], 1), 1.0)
    return np.linalg.pinv(a + ridge * scale * np.eye(a.shape[0]), rcond=1e-12)


def cov_stats(c: np.ndarray) -> Dict[str, float]:
    c = 0.5 * (np.asarray(c, dtype=np.float64) + np.asarray(c, dtype=np.float64).T)
    ev = np.linalg.eigvalsh(c)
    d = np.sqrt(np.maximum(np.diag(c), 1e-300))
    corr = float(c[0, 1] / max(d[0] * d[1], 1e-300)) if c.shape == (2, 2) else float("nan")
    det = float(np.linalg.det(c)) if c.shape == (2, 2) else float("nan")
    return {
        "trace": float(np.trace(c)),
        "det": det,
        "min_eig": float(np.min(ev)),
        "max_eig": float(np.max(ev)),
        "condition": float(np.max(ev) / max(np.min(ev), 1e-14)),
        "corr": corr,
    }


def find_array_attr(obj: Any, candidates: Sequence[str], ndim: int | None = None):
    for name in candidates:
        if hasattr(obj, name):
            val = np.asarray(getattr(obj, name), dtype=np.float64)
            if ndim is None or val.ndim == ndim:
                return name, val
    return None, None


def build_model_and_evaluator(args):
    d8 = load_module(Path(args.d8_script), "d8_mech")
    backend = load_module(Path(args.backend), "stage_b_mech")
    c2 = load_module(Path(args.c2_script), "stage_c2_mech")
    d2 = load_module(Path(args.d2_script), "stage_d2_mech")
    d3 = load_module(Path(args.d3_script), "stage_d3_mech")
    d5 = load_module(Path(args.d5_script), "stage_d5_mech")
    d7 = load_module(Path(args.d7_script), "stage_d7_mech")

    if hasattr(d8, "install_robust_d3_projection"):
        d8.install_robust_d3_projection(d3)

    params, checkpoint_meta = d5.load_checkpoint(Path(args.checkpoint))
    cfg = d8.preset_d8_config(args.preset)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=int(args.seed))

    base_b = backend.preset_config("quick" if args.preset == "quick" else "reference")
    stage_b_cfg = dataclasses.replace(base_b, grid_n=int(cfg.grid_n), time_n=int(cfg.time_n))
    model = backend.StageB(stage_b_cfg)
    d7_cfg = d8.d7_compatible_config(cfg, d7)
    evaluator = d7.D7Evaluator(model, d2, d5, params, checkpoint_meta, d7_cfg)
    measurement_cov = d3.MeasurementCovariance(model)

    acq_sets = c2.nested_acquisition_sets(model.cfg.time_n, [int(cfg.acquisition_k)])
    acq_idx = np.asarray(acq_sets[int(cfg.acquisition_k)], dtype=int)
    acq_set = set(acq_idx.tolist())
    heldout_idx = np.asarray([
        i for i in range(model.cfg.time_n)
        if i not in acq_set and i not in (0, model.cfg.time_n - 1)
    ], dtype=int)
    heldout_mask = np.zeros(model.cfg.time_n, dtype=bool)
    heldout_mask[heldout_idx] = True

    return d8, backend, c2, d2, d3, d5, d7, cfg, d7_cfg, model, evaluator, measurement_cov, acq_idx, heldout_idx, heldout_mask, checkpoint_meta


def exact_validation_bank(cfg, model, c2, acq_idx):
    total = int(cfg.law_trials + cfg.validation_trials)
    bank = []
    for trial in range(total):
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 8800, trial]))
        alpha = float(rng.uniform(model.cfg.alpha_min, model.cfg.alpha_max))
        bank.append(c2.draw_shared_trial(model, alpha, acq_idx, int(cfg.finite_n), rng))
    return bank[int(cfg.law_trials):]


def load_validation_rows(path: Path | None):
    if path is None or not Path(path).exists():
        return None, None
    payload = json.loads(Path(path).read_text())
    rows = payload.get("validation_trial_rows")
    if not isinstance(rows, dict):
        return payload, None
    if "robust_d8" not in rows or "full" not in rows:
        return payload, None
    return payload, {"d8": rows["robust_d8"], "full": rows["full"]}


def recompute_validation_rows(
    designs, model, evaluator, d2, d3, d7, c2, measurement_cov,
    cfg, d7_cfg, validation_bank, acq_idx, heldout_mask,
):
    out = {k: [] for k in designs}
    constraints = {}
    for name, eta in designs.items():
        constraints[name] = d7.build_joint_beta_constraints(
            model=model,
            evaluator=evaluator,
            d2=d2,
            c2=c2,
            eta=eta,
            margin=float(cfg.feasibility_margin),
        )
    for j, shared in enumerate(validation_bank):
        for name, eta in designs.items():
            row, _ = d7.evaluate_design_trial(
                model=model,
                evaluator=evaluator,
                d2=d2,
                d3=d3,
                c2=c2,
                measurement_cov=measurement_cov,
                eta=eta,
                shared=shared,
                acq_idx=acq_idx,
                heldout_mask=heldout_mask,
                cfg=d7_cfg,
                joint_constraints=constraints[name],
                compute_action=True,
            )
            out[name].append(row)
        print(f"  validation pair {j+1:2d}/{len(validation_bank)} alpha={shared.alpha:.5f}", flush=True)
    return out


def mean_se(x: Sequence[float]) -> Dict[str, float | int]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(a)),
        "se": float(np.std(a, ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0,
        "n": int(a.size),
    }


def paired_difference(rows_a, rows_b, key):
    a = np.asarray([r.get(key, np.nan) for r in rows_a], dtype=np.float64)
    b = np.asarray([r.get(key, np.nan) for r in rows_b], dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    s = mean_se(d)
    return {
        "mean_a_minus_b": s["mean"],
        "se": s["se"],
        "n": s["n"],
        "a_lower_fraction": float(np.mean(d < 0.0)) if d.size else float("nan"),
    }


def paired_reduction(rows_num, rows_den, key):
    num = np.asarray([r.get(key, np.nan) for r in rows_num], dtype=np.float64)
    den = np.asarray([r.get(key, np.nan) for r in rows_den], dtype=np.float64)
    keep = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-14)
    num, den = num[keep], den[keep]
    if num.size == 0:
        return {"ratio_of_means_reduction": float("nan"), "mean_paired_reduction": float("nan"), "se_paired_reduction": float("nan"), "n": 0}
    pr = 1.0 - num / den
    s = mean_se(pr)
    return {
        "ratio_of_means_reduction": float(1.0 - np.mean(num) / np.mean(den)),
        "mean_paired_reduction": s["mean"],
        "se_paired_reduction": s["se"],
        "n": s["n"],
    }


def action_distribution(rows, key="finite_action"):
    x = np.asarray([r.get(key, np.nan) for r in rows], dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
        "cv": float(np.std(x, ddof=1) / max(abs(np.mean(x)), 1e-14)) if x.size > 1 else 0.0,
        "n": int(x.size),
    }


def reference_geometry_profiles(model, evaluator, d2, d8, designs):
    node_name, nodes = find_array_attr(
        evaluator,
        ("learned_nodes", "nodes", "xnodes", "x_nodes", "reference_nodes", "particle_nodes"),
        ndim=3,
    )
    vel_name, unodes = find_array_attr(
        evaluator,
        ("learned_u_nodes", "u_nodes", "velocity_nodes", "vel_nodes", "reference_u_nodes"),
        ndim=3,
    )
    base_name, base_w = find_array_attr(
        evaluator,
        ("base_w", "base_weights", "weights0", "quadrature_weights"),
        ndim=1,
    )
    if nodes is None or unodes is None or base_w is None:
        return {
            "available": False,
            "reason": "Could not locate learned nodes, learned velocities and base weights on D7Evaluator.",
            "discovered_attributes": {"nodes": node_name, "velocities": vel_name, "base_weights": base_name},
        }

    times = np.asarray(model.times, dtype=np.float64)
    tw = np.asarray(model.time_w, dtype=np.float64)
    alphas = np.asarray(model.alphas, dtype=np.float64)
    aw = np.asarray(model.alpha_w, dtype=np.float64)
    aw = aw / np.sum(aw)
    tw = tw / np.sum(tw)
    out = {
        "available": True,
        "discovered_attributes": {"nodes": node_name, "velocities": vel_name, "base_weights": base_name},
        "designs": {},
    }

    for name, eta in designs.items():
        phi_grid, _ = model.sensor_fields(jnp.asarray(eta, dtype=jnp.float64))
        phi_grid = np.asarray(phi_grid, dtype=np.float64)
        per_time = []
        accum = {
            "raw_moment_gap_sq": 0.0,
            "raw_moment_mahal": 0.0,
            "zero_tilt_dynamic_mismatch_sq": 0.0,
            "zero_tilt_tangent_proxy": 0.0,
            "target_speed_sq": 0.0,
            "ref_sensor_cov_trace": 0.0,
            "external_sensor_cov_trace": 0.0,
            "abs_ref_sensor_corr": 0.0,
            "abs_external_sensor_corr": 0.0,
            "alpha_sensitivity_trace": 0.0,
            "alpha_sensitivity_det": 0.0,
        }

        # Precompute exact target c/cdot for every scientific alpha.
        curves = []
        for alpha in alphas:
            c, cdot = d8.exact_target_curve(model, evaluator, eta, float(alpha))
            curves.append((c, cdot))

        for kt, t in enumerate(times):
            x_all = np.asarray(nodes[kt], dtype=np.float64)
            u_all = np.asarray(unodes[kt], dtype=np.float64)
            mask = d2.in_domain_mask(model, x_all)
            x = x_all[mask]
            u = u_all[mask]
            w = np.asarray(base_w[mask], dtype=np.float64)
            w /= max(float(np.sum(w)), 1e-300)
            phi, grad_phi = d2.sensor_particle_fields(model, eta, x)
            ref_mu = np.sum(w[:, None] * phi, axis=0)
            Cref = weighted_cov(w, phi)
            Cref_inv = sym_pinv(Cref)
            m = np.einsum("nmc,nc->nm", grad_phi, u)
            ref_mdot = np.sum(w[:, None] * m, axis=0)
            G = np.einsum("nmc,nkc,n->mk", grad_phi, grad_phi, w)
            Ginv = sym_pinv(G)
            ref_cs = cov_stats(Cref)

            # Between-alpha sensitivity of the scientific moment target at this time.
            c_alpha = np.stack([curves[ka][0][kt] for ka in range(len(alphas))], axis=0)
            calpha_cov = weighted_cov(aw, c_alpha)
            calpha_cs = cov_stats(calpha_cov)

            vals = {
                "raw_moment_gap_sq": 0.0,
                "raw_moment_mahal": 0.0,
                "zero_tilt_dynamic_mismatch_sq": 0.0,
                "zero_tilt_tangent_proxy": 0.0,
                "target_speed_sq": 0.0,
                "external_sensor_cov_trace": 0.0,
                "abs_external_sensor_corr": 0.0,
            }
            external_min_eigs = []
            external_dets = []

            for ka, alpha in enumerate(alphas):
                c = curves[ka][0][kt]
                cdot = curves[ka][1][kt]
                gap = c - ref_mu
                dyn = cdot - ref_mdot
                vals["raw_moment_gap_sq"] += aw[ka] * float(gap @ gap)
                vals["raw_moment_mahal"] += aw[ka] * float(gap @ Cref_inv @ gap)
                vals["zero_tilt_dynamic_mismatch_sq"] += aw[ka] * float(dyn @ dyn)
                vals["zero_tilt_tangent_proxy"] += aw[ka] * float(dyn @ Ginv @ dyn)
                vals["target_speed_sq"] += aw[ka] * float(cdot @ cdot)

                _, pmass = model.external_q_mass(jnp.asarray(t), jnp.asarray(alpha))
                pmass = np.asarray(pmass, dtype=np.float64)
                mu_ext = np.sum(phi_grid * pmass[None, ...], axis=(1, 2))
                centered = phi_grid - mu_ext[:, None, None]
                Cext = np.einsum("myx,nyx,yx->mn", centered, centered, pmass)
                ecs = cov_stats(Cext)
                vals["external_sensor_cov_trace"] += aw[ka] * ecs["trace"]
                vals["abs_external_sensor_corr"] += aw[ka] * abs(ecs["corr"])
                external_min_eigs.append(ecs["min_eig"])
                external_dets.append(ecs["det"])

            row = {
                "t": float(t),
                **{k: float(v) for k, v in vals.items()},
                "raw_moment_gap_rms": float(math.sqrt(max(vals["raw_moment_gap_sq"], 0.0))),
                "zero_tilt_dynamic_mismatch_rms": float(math.sqrt(max(vals["zero_tilt_dynamic_mismatch_sq"], 0.0))),
                "ref_sensor_cov_trace": ref_cs["trace"],
                "ref_sensor_cov_min_eig": ref_cs["min_eig"],
                "ref_sensor_cov_det": ref_cs["det"],
                "ref_sensor_corr": ref_cs["corr"],
                "external_sensor_cov_min_eig_mean": float(np.sum(aw * np.asarray(external_min_eigs))),
                "external_sensor_cov_det_mean": float(np.sum(aw * np.asarray(external_dets))),
                "alpha_sensitivity_trace": calpha_cs["trace"],
                "alpha_sensitivity_det": calpha_cs["det"],
                "alpha_sensitivity_min_eig": calpha_cs["min_eig"],
            }
            per_time.append(row)

            for k in accum:
                if k == "ref_sensor_cov_trace":
                    v = ref_cs["trace"]
                elif k == "abs_ref_sensor_corr":
                    v = abs(ref_cs["corr"])
                elif k == "alpha_sensitivity_trace":
                    v = calpha_cs["trace"]
                elif k == "alpha_sensitivity_det":
                    v = calpha_cs["det"]
                else:
                    v = vals[k]
                accum[k] += tw[kt] * float(v)

        out["designs"][name] = {
            "eta_deg": np.degrees(eta).tolist(),
            "time_integrated": {
                **{k: float(v) for k, v in accum.items()},
                "raw_moment_gap_rms": float(math.sqrt(max(accum["raw_moment_gap_sq"], 0.0))),
                "zero_tilt_dynamic_mismatch_rms": float(math.sqrt(max(accum["zero_tilt_dynamic_mismatch_sq"], 0.0))),
            },
            "per_time": per_time,
        }
    return out


def paired_trial_table(rows_d8, rows_full):
    keys = [
        "alpha", "finite_heldout_mmd2", "exact_heldout_mmd2", "measurement_delta_mmd2",
        "finite_action", "exact_action", "measurement_action_inflation",
        "finite_min_ess", "finite_min_chi2_ess", "finite_max_calibration_resid",
        "finite_max_abs_lambda_coordinate", "finite_max_clip_fraction", "finite_retry_case_count",
        "feasibility_projection_active", "feasibility_projection_norm",
        "quadratic_moment_rmse", "quadratic_moment_max_error", "acquisition_mean_rmse",
    ]
    out = []
    for i, (a, b) in enumerate(zip(rows_d8, rows_full)):
        r = {"trial_1based": i + 1}
        for k in keys:
            r[f"d8_{k}"] = a.get(k, np.nan)
            r[f"full_{k}"] = b.get(k, np.nan)
        if np.isfinite(float(a.get("finite_action", np.nan))) and np.isfinite(float(b.get("finite_action", np.nan))) and abs(float(b.get("finite_action", 0.0))) > 1e-14:
            r["paired_action_reduction_d8_vs_full"] = 1.0 - float(a["finite_action"]) / float(b["finite_action"])
        if np.isfinite(float(a.get("finite_heldout_mmd2", np.nan))) and np.isfinite(float(b.get("finite_heldout_mmd2", np.nan))):
            r["paired_finite_law_d8_minus_full"] = float(a["finite_heldout_mmd2"] - b["finite_heldout_mmd2"])
        out.append(r)
    return out


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def build_interpretation(pop, summaries, paired, geom):
    notes = []
    d8p, fp = pop["d8"], pop["full"]
    if np.isfinite(d8p.get("lift_mmd2", np.nan)) and np.isfinite(fp.get("lift_mmd2", np.nan)):
        rel = d8p["lift_mmd2"] / fp["lift_mmd2"] - 1.0
        notes.append(
            f"Exact-population projected-law MMD: D8 is {100*rel:+.2f}% relative to Full. "
            "If this is already strongly negative, the law advantage is not created by finite measurement noise."
        )
    raw = d8p.get("raw_lift_mmd2", np.nan)
    if np.isfinite(raw):
        gd8 = raw - d8p.get("lift_mmd2", np.nan)
        gf = fp.get("raw_lift_mmd2", raw) - fp.get("lift_mmd2", np.nan)
        notes.append(
            f"I-projection improvement over raw D5 reference: D8={gd8:+.4e}, Full={gf:+.4e}. "
            "A larger D8 improvement means its measurements pull the same D5 reference more strongly toward the external law."
        )
    fd = paired.get("finite_action_d8_vs_full_reduction", {})
    if np.isfinite(fd.get("ratio_of_means_reduction", np.nan)):
        notes.append(
            f"D8-vs-Full finite-action reduction is {100*fd['ratio_of_means_reduction']:+.2f}% by ratio of means; "
            "negative means D8 is MORE expensive. Low action alone therefore does not imply better law fidelity."
        )
    if geom.get("available"):
        gd = geom["designs"]["d8"]["time_integrated"]
        gf = geom["designs"]["full"]["time_integrated"]
        if gf["alpha_sensitivity_trace"] > 0:
            notes.append(
                f"Between-alpha target-moment sensitivity trace ratio D8/Full={gd['alpha_sensitivity_trace']/gf['alpha_sensitivity_trace']:.3f}. "
                "Values >1 support the hypothesis that D8 measures the scientifically varying alpha direction more strongly."
            )
        if gf["raw_moment_mahal"] > 0:
            notes.append(
                f"Raw-reference standardized moment-mismatch ratio D8/Full={gd['raw_moment_mahal']/gf['raw_moment_mahal']:.3f}. "
                "Values >1 mean D8 more clearly detects that the raw D5 reference is off-fiber relative to the external scientific law."
            )
        if gf["zero_tilt_tangent_proxy"] > 0:
            notes.append(
                f"Zero-tilt dynamic-burden proxy ratio D8/Full={gd['zero_tilt_tangent_proxy']/gf['zero_tilt_tangent_proxy']:.3f}. "
                "This is only an upstream diagnostic; the authoritative realization cost remains the weighted-Poisson action."
            )
    return notes


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d8-script", default="stage_d8_endpoint_flow_matching_finite_resource_design_oracle_fixed.py")
    p.add_argument("--backend", required=True)
    p.add_argument("--c2-script", required=True)
    p.add_argument("--d2-script", required=True)
    p.add_argument("--d3-script", required=True)
    p.add_argument("--d5-script", required=True)
    p.add_argument("--d7-script", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="reference")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--selected-eta-deg", type=float, nargs=2, default=(19.45945945945946, 68.10810810810811))
    p.add_argument("--full-eta-deg", type=float, nargs=2, default=None)
    p.add_argument("--validation-json", default="stage_d8_endpoint_flow_matching_finite_resource_design_oracle_validation_resume.json")
    p.add_argument("--force-validation-recompute", action="store_true")
    p.add_argument("--output", default="stage_d8b_d8_vs_full_mechanism.json")
    return p


def main():
    wall0 = time.time()
    args = build_parser().parse_args()
    (
        d8, backend, c2, d2, d3, d5, d7, cfg, d7_cfg, model, evaluator,
        measurement_cov, acq_idx, heldout_idx, heldout_mask, checkpoint_meta,
    ) = build_model_and_evaluator(args)

    full_deg = args.full_eta_deg if args.full_eta_deg is not None else cfg.full_design_deg
    designs = {
        "d8": d8.canonical_eta(np.radians(np.asarray(args.selected_eta_deg, dtype=np.float64))),
        "full": d8.canonical_eta(np.radians(np.asarray(full_deg, dtype=np.float64))),
    }

    print("=" * 112)
    print("Stage D.8b — D8 vs historical Full-TC mechanism diagnostic")
    print("=" * 112)
    print(f"D8 eta    : {np.degrees(designs['d8']).round(6).tolist()} deg")
    print(f"Full eta  : {np.degrees(designs['full']).round(6).tolist()} deg")
    print(f"Grid/time : {model.cfg.grid_n}x{model.cfg.grid_n} / {model.cfg.time_n}")
    print(f"Validation: exact D8 seed offset law_trials={cfg.law_trials}, n={cfg.validation_trials}")

    # ---------------------------------------------------------------
    # A. Population-law geometry under the NEW D5 reference.
    # ---------------------------------------------------------------
    print("\n[A] Exact-population D5-reference law diagnostics...", flush=True)
    population = {}
    for name, eta in designs.items():
        population[name] = d8.population_endpoint_fm_law(
            model=model, evaluator=evaluator, d2=d2, d7=d7, eta=eta, cfg=cfg
        )
        p = population[name]
        print(
            f"  {name:5s}: projected L={p['lift_mmd2']:.8e} | raw L={p['raw_lift_mmd2']:.8e} | "
            f"ESSmin={p['min_ess_fraction']:.3f} | max|lambda|={p['max_abs_lambda_coordinate']:.3f}",
            flush=True,
        )

    # ---------------------------------------------------------------
    # B. Exact same independent D8 validation rows.
    # ---------------------------------------------------------------
    print("\n[B] Paired independent-validation diagnostics...", flush=True)
    source_payload, validation_rows = (None, None)
    vpath = Path(args.validation_json) if args.validation_json else None
    if not args.force_validation_recompute:
        source_payload, validation_rows = load_validation_rows(vpath)
    if validation_rows is not None:
        print(f"  Reusing trial rows from {vpath.resolve()}")
    else:
        print("  Validation rows unavailable; recomputing only D8 + Full on exact original D8 validation bank.")
        validation_bank = exact_validation_bank(cfg, model, c2, acq_idx)
        validation_rows = recompute_validation_rows(
            designs, model, evaluator, d2, d3, d7, c2, measurement_cov,
            cfg, d7_cfg, validation_bank, acq_idx, heldout_mask,
        )

    summaries = {
        name: d8.summarize_selection_rows(rows)
        for name, rows in validation_rows.items()
    }
    paired = {
        "finite_law_d8_minus_full": paired_difference(validation_rows["d8"], validation_rows["full"], "finite_heldout_mmd2"),
        "exact_law_d8_minus_full": paired_difference(validation_rows["d8"], validation_rows["full"], "exact_heldout_mmd2"),
        "measurement_degradation_d8_minus_full": paired_difference(validation_rows["d8"], validation_rows["full"], "measurement_delta_mmd2"),
        "finite_action_d8_vs_full_reduction": paired_reduction(validation_rows["d8"], validation_rows["full"], "finite_action"),
        "exact_action_d8_vs_full_reduction": paired_reduction(validation_rows["d8"], validation_rows["full"], "exact_action"),
        "finite_min_ess_d8_minus_full": paired_difference(validation_rows["d8"], validation_rows["full"], "finite_min_ess"),
        "finite_lambda_d8_minus_full": paired_difference(validation_rows["d8"], validation_rows["full"], "finite_max_abs_lambda_coordinate"),
        "quadratic_rmse_d8_minus_full": paired_difference(validation_rows["d8"], validation_rows["full"], "quadratic_moment_rmse"),
        "projection_norm_d8_minus_full": paired_difference(validation_rows["d8"], validation_rows["full"], "feasibility_projection_norm"),
    }
    distributions = {
        "d8_finite_action": action_distribution(validation_rows["d8"], "finite_action"),
        "full_finite_action": action_distribution(validation_rows["full"], "finite_action"),
        "d8_exact_action": action_distribution(validation_rows["d8"], "exact_action"),
        "full_exact_action": action_distribution(validation_rows["full"], "exact_action"),
    }
    trial_pairs = paired_trial_table(validation_rows["d8"], validation_rows["full"])
    worst_full = sorted(
        trial_pairs,
        key=lambda r: float(r.get("full_finite_action", -np.inf)) if np.isfinite(float(r.get("full_finite_action", np.nan))) else -np.inf,
        reverse=True,
    )[:5]

    print(
        f"  finite law: D8={summaries['d8']['finite_heldout_mmd2']['mean']:.8e}, "
        f"Full={summaries['full']['finite_heldout_mmd2']['mean']:.8e}"
    )
    print(
        f"  exact law : D8={summaries['d8']['exact_heldout_mmd2']['mean']:.8e}, "
        f"Full={summaries['full']['exact_heldout_mmd2']['mean']:.8e}"
    )
    print(
        f"  action    : D8={summaries['d8']['finite_action']['mean']:.6e}, "
        f"Full={summaries['full']['finite_action']['mean']:.6e}"
    )

    # ---------------------------------------------------------------
    # C. Upstream measurement/reference geometry.
    # ---------------------------------------------------------------
    print("\n[C] Raw D5-reference / measurement-geometry diagnostics...", flush=True)
    geometry = reference_geometry_profiles(model, evaluator, d2, d8, designs)
    if geometry.get("available"):
        for name in ("d8", "full"):
            g = geometry["designs"][name]["time_integrated"]
            print(
                f"  {name:5s}: alpha-sens trace={g['alpha_sensitivity_trace']:.6e} | "
                f"raw mismatch mahal={g['raw_moment_mahal']:.6e} | "
                f"zero-tilt dyn proxy={g['zero_tilt_tangent_proxy']:.6e} | "
                f"|corr_ref|={g['abs_ref_sensor_corr']:.3f}"
            )
    else:
        print(f"  Skipped: {geometry.get('reason')}")

    # ---------------------------------------------------------------
    # D. Save machine-readable attribution package + CSVs.
    # ---------------------------------------------------------------
    interpretation = build_interpretation(population, summaries, paired, geometry)
    output = Path(args.output)
    trial_csv = output.with_suffix(".trial_pairs.csv")
    time_csv = output.with_suffix(".time_profiles.csv")
    write_csv(trial_csv, trial_pairs)
    if geometry.get("available"):
        time_rows = []
        for name in ("d8", "full"):
            for r in geometry["designs"][name]["per_time"]:
                time_rows.append({"design": name, **r})
        write_csv(time_csv, time_rows)

    payload = {
        "stage": "D.8b D8-vs-Full mechanism diagnostic",
        "purpose": "Explain why the D8 sensor has much better law fidelity than historical Full-TC while Full-TC has lower realization action under the D5 endpoint-trained reference.",
        "designs": {
            "d8": {"eta_deg": np.degrees(designs["d8"]).tolist(), "eta_rad": designs["d8"]},
            "full": {"eta_deg": np.degrees(designs["full"]).tolist(), "eta_rad": designs["full"]},
        },
        "historical_context": {
            "full_tc_is_historical_stage_b_design": True,
            "full_tc_was_not_reoptimized_for_d5": True,
            "d8_was_reoptimized_for_d5_finite_resource_pipeline": True,
        },
        "config": jsonify(cfg),
        "d7_config": jsonify(d7_cfg),
        "checkpoint_metadata": checkpoint_meta,
        "condition": {
            "N": int(cfg.finite_n),
            "K": int(cfg.acquisition_k),
            "obs_noise_std": float(cfg.obs_noise_std),
            "validation_trials": int(cfg.validation_trials),
            "heldout_indices": heldout_idx.tolist(),
        },
        "population_exact": population,
        "validation_source_json": str(vpath.resolve()) if vpath is not None and vpath.exists() else None,
        "validation_recomputed": bool(source_payload is None or validation_rows is None),
        "validation_summary": summaries,
        "paired_validation": paired,
        "action_distributions": distributions,
        "worst_full_action_trials": worst_full,
        "reference_measurement_geometry": geometry,
        "interpretation": interpretation,
        "important_caveats": [
            "The raw-moment Mahalanobis and zero-tilt tangent quantities are mechanism diagnostics, not substitutes for the exact I-projected weighted-Poisson action.",
            "Full-TC is a historical optimizer from the old Stage-B reference geometry; D8 is optimized under the new endpoint-trained D5 reference and finite-resource screens.",
            "Finite-law and exact-law comparisons use the same D5 reference. A persistent exact-law gap attributes the difference upstream of finite measurement noise.",
            "The validation comparison is paired by the original D8 CRN trial index.",
        ],
        "csv_outputs": {
            "trial_pairs": str(trial_csv),
            "time_profiles": str(time_csv) if geometry.get("available") else None,
        },
        "wall_seconds": float(time.time() - wall0),
    }
    atomic_json(output, payload)

    print("\n" + "=" * 112)
    print("Mechanism readout")
    print("=" * 112)
    for note in interpretation:
        print("- " + note)
    print(f"\nSaved JSON       : {output.resolve()}")
    print(f"Saved trial CSV  : {trial_csv.resolve()}")
    if geometry.get("available"):
        print(f"Saved time CSV   : {time_csv.resolve()}")


if __name__ == "__main__":
    main()
