from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.design import (
    OptimizeResult,
    OptimizerConfig,
    optimize_multistart_candidates,
    point_box_violation,
    point_separation_violation,
)

Array = jax.Array


def _prefix_bank(bank, count: int):
    count = max(1, min(int(count), int(bank.sample_indices.shape[0])))
    return type(bank)(bank.sample_indices[:count], bank.detector_z[:count])


def _opt_cfg(cfg: dict[str, Any], prefix: str) -> OptimizerConfig:
    o = cfg["optimization"]
    return OptimizerConfig(
        steps=int(o.get(f"{prefix}_steps", o.get("steps", 100))),
        learning_rate=float(o.get(f"{prefix}_learning_rate", o.get("learning_rate", 0.01))),
        beta1=float(o.get("beta1", 0.9)), beta2=float(o.get("beta2", 0.999)),
        eps=float(o.get("eps", 1.0e-8)),
        constraint_penalty=float(o.get("constraint_penalty", 1.0e4)),
        feasibility_tol=float(o.get("feasibility_tol", 1.0e-6)),
    )


def _key(exp, eta: Array) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(exp.family.canonicalize(eta), dtype=np.float64), 12))


def _dedupe(exp, values) -> list[Array]:
    out: dict[tuple[float, ...], Array] = {}
    for value in values:
        eta = value.eta if isinstance(value, OptimizeResult) else value
        eta = exp.family.canonicalize(jnp.asarray(eta, dtype=jnp.float64))
        out[_key(exp, eta)] = eta
    return list(out.values())


def _geometry_constraints(exp):
    m = exp.cfg["measurement"]
    margin = float(m.get("boundary_margin", 2.0 * float(m.get("sensor_width", 0.12))))
    xb = (exp.grid.x_min + margin, exp.grid.x_max - margin)
    yb = (exp.grid.y_min + margin, exp.grid.y_max - margin)
    return (
        (point_box_violation(n_sensors=exp.family.n_sensors, x_bounds=xb, y_bounds=yb), 0.0),
        (point_separation_violation(float(m.get("min_sep", 0.24)), n_sensors=exp.family.n_sensors), 0.0),
    )


def _local_cloud(exp, centers: list[Array], *, count_per_center: int, scale: float, seed: int) -> list[Array]:
    if count_per_center <= 0:
        return []
    m = exp.cfg["measurement"]
    margin = float(m.get("boundary_margin", 2.0 * float(m.get("sensor_width", 0.12))))
    lo = np.asarray([exp.grid.x_min + margin, exp.grid.y_min + margin], dtype=np.float64)
    hi = np.asarray([exp.grid.x_max - margin, exp.grid.y_max - margin], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    out = []
    for eta in _dedupe(exp, centers):
        c = np.asarray(exp.family.centers(eta), dtype=np.float64)
        for k in range(int(count_per_center)):
            # Multiscale perturbations deliberately include very local candidates.
            frac = (k + 1.0) / float(count_per_center)
            step = float(scale) * frac * frac
            cand = np.clip(c + step * rng.standard_normal(c.shape), lo, hi)
            out.append(jnp.asarray(cand.reshape(-1), dtype=jnp.float64))
    return _dedupe(exp, out)


def _rank_pool(exp, pool: list[Array], score: Callable[[Array], Array], constraints) -> list[tuple[float, float, Array]]:
    score_jit = jax.jit(score)
    ranked = []
    for eta in _dedupe(exp, pool):
        violation = max([0.0] + [float(fn(eta) - upper) for fn, upper in constraints])
        ranked.append((violation, float(score_jit(eta)), eta))
    ranked.sort(key=lambda r: (r[0], r[1]))
    return ranked


def _audit_population(exp, ranked, *, limit: int, min_valid: int) -> tuple[Array, list[dict[str, Any]]]:
    # boundary-fix diagnostic version: keep the authoritative acceptance rule,
    # but surface why candidates were rejected instead of collapsing everything
    # into an opaque "no valid candidate" exception.
    rows = []
    validity = exp.cfg.get("validity", {})
    max_resid_allowed = float(validity.get("max_population_calibration_resid", 1.0e-5))
    min_ess_allowed = float(validity.get("min_ess_fraction", 0.03))
    min_base_mass_allowed = float(validity.get("min_in_domain_base_mass", 0.995))
    reference_min_base_mass = float(np.min(np.asarray(exp.reference_base_mass, dtype=np.float64)))

    for _, fast, eta in ranked:
        if len(rows) >= int(limit) and sum(r["valid"] for r in rows) >= int(min_valid):
            break
        exact = exp.exact_population_result(eta)
        resid = exact.get("max_calibration_residual")
        ess = exact.get("min_ess_fraction")
        resid_f = float(resid) if resid is not None else float("nan")
        ess_f = float(ess) if ess is not None else float("nan")
        exact_value = float(exact["value"])

        reasons = []
        if not np.isfinite(exact_value):
            reasons.append("nonfinite_exact_risk")
        if not np.isfinite(resid_f) or resid_f > max_resid_allowed:
            reasons.append("calibration_residual")
        if not np.isfinite(ess_f) or ess_f < min_ess_allowed:
            reasons.append("ess")
        if reference_min_base_mass < min_base_mass_allowed:
            reasons.append("reference_in_domain_mass")
        if not bool(exact["valid"]) and not reasons:
            reasons.append("other_exact_validity_gate")

        rows.append({
            "eta": np.asarray(eta).tolist(),
            "fast": float(fast),
            "exact": exact_value,
            "valid": bool(exact["valid"]),
            "max_calibration_residual": resid_f,
            "min_ess_fraction": ess_f,
            "rejection_reasons": reasons,
        })

    valid = [r for r in rows if r["valid"] and np.isfinite(r["exact"])]
    if not valid:
        finite_resid = [r["max_calibration_residual"] for r in rows if np.isfinite(r["max_calibration_residual"])]
        finite_ess = [r["min_ess_fraction"] for r in rows if np.isfinite(r["min_ess_fraction"])]
        counts: dict[str, int] = {}
        for row in rows:
            for reason in row["rejection_reasons"]:
                counts[reason] = counts.get(reason, 0) + 1
        raise RuntimeError(
            "no exact-valid DG-Exact population candidate; "
            f"audited={len(rows)}, exact_valid=0, "
            f"reference_min_in_domain_mass={reference_min_base_mass:.8f} "
            f"(required>={min_base_mass_allowed:.8f}), "
            f"best_calibration_residual={min(finite_resid) if finite_resid else float('nan'):.3e} "
            f"(required<={max_resid_allowed:.3e}), "
            f"best_min_ess={max(finite_ess) if finite_ess else float('nan'):.6f} "
            f"(required>={min_ess_allowed:.6f}), "
            f"rejection_counts={counts}"
        )
    best = min(valid, key=lambda r: r["exact"])
    return jnp.asarray(best["eta"], dtype=jnp.float64), rows




def _audit_law(
    exp,
    ranked,
    law_bank,
    *,
    L_max: float,
    limit: int,
    min_valid: int,
    mandatory: list[Array] | None = None,
) -> tuple[Array, list[dict[str, Any]]]:
    mandatory_keys = {_key(exp, e) for e in (mandatory or [])}
    ordered = [r for r in ranked if _key(exp, r[2]) in mandatory_keys] + [r for r in ranked if _key(exp, r[2]) not in mandatory_keys]
    rows = []
    for violation, fast, eta in ordered:
        if len(rows) >= int(limit) and sum(r["valid"] for r in rows) >= int(min_valid):
            break
        pop = exp.exact_population_result(eta)
        if not pop["valid"] or float(pop["value"]) > float(L_max) + 1.0e-12:
            rows.append({"eta": np.asarray(eta).tolist(), "fast": fast, "fast_violation": violation,
                         "exact_L": float(pop["value"]), "exact_R": float("inf"), "valid": False})
            continue
        fin = exp.exact_finite_result(eta, law_bank)
        rows.append({"eta": np.asarray(eta).tolist(), "fast": fast, "fast_violation": violation,
                     "exact_L": float(pop["value"]), "exact_R": float(fin["value"]), "valid": bool(fin["valid"])})
    valid = [r for r in rows if r["valid"] and np.isfinite(r["exact_R"])]
    if not valid:
        raise RuntimeError("no exact-valid finite-observation law candidate")
    best = min(valid, key=lambda r: r["exact_R"])
    return jnp.asarray(best["eta"], dtype=jnp.float64), rows


def _audit_action(
    name: str,
    exp,
    ranked,
    law_bank,
    action_bank,
    *,
    L_max: float,
    R_max: float,
    exact_action,
    exact_prescreen=None,
    audit_limit: int,
    finalist_count: int,
    mandatory: list[Array] | None = None,
) -> tuple[Array, list[dict[str, Any]]]:
    mandatory_keys = {_key(exp, e) for e in (mandatory or [])}
    ordered = [r for r in ranked if _key(exp, r[2]) in mandatory_keys] + [r for r in ranked if _key(exp, r[2]) not in mandatory_keys]
    law_valid = []
    audited = 0
    for violation, proxy, eta in ordered:
        if audited >= int(audit_limit) and len(law_valid) >= int(finalist_count):
            break
        audited += 1
        pop = exp.exact_population_result(eta)
        if not pop["valid"] or float(pop["value"]) > float(L_max) + 1.0e-12:
            continue
        fin = exp.exact_finite_result(eta, law_bank)
        if not fin["valid"] or float(fin["value"]) > float(R_max) + 1.0e-12:
            continue
        law_valid.append({
            "eta": eta, "proxy": float(proxy), "fast_violation": float(violation),
            "exact_L": float(pop["value"]), "exact_R": float(fin["value"]),
        })
    if not law_valid:
        raise RuntimeError(f"no exact law-feasible {name} candidate")

    if exact_prescreen is not None:
        prescreened = []
        for row in law_valid:
            rec = exact_prescreen(row["eta"])
            if rec["valid"] and np.isfinite(rec["value"]):
                rr = dict(row)
                rr["prescreen"] = float(rec["value"])
                prescreened.append(rr)
        if not prescreened:
            raise RuntimeError(f"no valid {name} candidate survived exact pre-screen")
        prescreened.sort(key=lambda r: r["prescreen"])
        finalists = prescreened[: max(1, min(int(finalist_count), len(prescreened)))]
        by_key = {_key(exp, r["eta"]): r for r in prescreened}
    else:
        ordered_valid = sorted(law_valid, key=lambda r: r["proxy"])
        finalists = ordered_valid[: max(1, min(int(finalist_count), len(ordered_valid)))]
        by_key = {_key(exp, r["eta"]): r for r in ordered_valid}
    final_keys = {_key(exp, r["eta"]) for r in finalists}
    for key in mandatory_keys:
        if key in by_key and key not in final_keys:
            finalists.append(by_key[key])
            final_keys.add(key)

    rows = []
    for row in finalists:
        rec = exact_action(row["eta"])
        rows.append({
            "eta": np.asarray(row["eta"]).tolist(), "proxy": row["proxy"],
            "exact_L": row["exact_L"], "exact_R": row["exact_R"],
            "prescreen": row.get("prescreen"), "objective": float(rec["value"]),
            "valid": bool(rec["valid"]),
        })
    valid = [r for r in rows if r["valid"] and np.isfinite(r["objective"])]
    if not valid:
        raise RuntimeError(f"no scientifically valid {name} finalist")
    best = min(valid, key=lambda r: r["objective"])
    return jnp.asarray(best["eta"], dtype=jnp.float64), rows


def optimize_vortex_designs(exp, law_bank, action_bank, starts: Array, output_dir: Path) -> dict[str, Any]:
    """Four-stage information/transport selection with authoritative exact rescoring.

    Search uses differentiable empirical projections; every selected design is decided
    by the robust non-differentiated I-projection and full declared action/risk banks.
    """
    del output_dir  # reserved for future per-stage cache files; scientific outputs live in experiment.py
    cfg = exp.cfg
    law_cfg = cfg["law"]
    opt = cfg["optimization"]
    eps_l = float(law_cfg["epsilon_l"])
    eps_r = float(law_cfg["epsilon_r"])
    geometry = _geometry_constraints(exp)

    # ------------------------------------------------------------------
    # 1/4 DG-Exact information risk L
    # ------------------------------------------------------------------
    print("[1/4] optimizing exact-moment oracle law risk L", flush=True)
    pop_cfg = _opt_cfg(cfg, "population")
    pop_candidates = optimize_multistart_candidates(
        exp.population_loss, starts, pop_cfg, constraints=geometry,
        canonicalize=exp.family.canonicalize,
    ) if pop_cfg.steps > 0 else []
    pop_pool = _dedupe(exp, list(starts) + [r.eta for r in pop_candidates])
    pop_ranked = _rank_pool(exp, pop_pool, exp.population_loss, geometry)
    population_eta, population_rows = _audit_population(
        exp, pop_ranked,
        limit=int(opt.get("population_exact_audit_candidates", 16)),
        min_valid=int(opt.get("population_min_exact_valid", 4)),
    )
    L_star = float(exp.exact_population_result(population_eta)["value"])
    L_max = L_star + eps_l

    # ------------------------------------------------------------------
    # 2/4 sparse/noisy spline law risk R under L tolerance
    # ------------------------------------------------------------------
    law_grad_bank = _prefix_bank(law_bank, int(opt.get("law_gradient_trials", min(4, int(law_bank.sample_indices.shape[0])))))
    fast_L_anchor = float(jax.jit(exp.population_loss)(population_eta))
    fast_L_max = fast_L_anchor + eps_l
    l_scale = max(eps_l, 1.0e-6 * max(abs(fast_L_anchor), 1.0), 1.0e-10)
    population_slack = lambda eta: (exp.population_loss(eta) - fast_L_max) / l_scale
    law_constraints = geometry + ((population_slack, 0.0),)
    law_obj = lambda eta: exp.finite_risk(eta, law_grad_bank)
    law_cfg_opt = _opt_cfg(cfg, "law")
    law_starts = jnp.stack(_dedupe(exp, [population_eta] + list(starts)))
    print("[2/4] optimizing sparse/noisy cubic-spline law risk R", flush=True)
    law_candidates = optimize_multistart_candidates(
        law_obj, law_starts, law_cfg_opt, constraints=law_constraints,
        canonicalize=exp.family.canonicalize,
    
        vectorize_starts=False,) if law_cfg_opt.steps > 0 else []
    law_pool = _dedupe(exp, list(law_starts) + [r.eta for r in law_candidates])
    law_ranked = _rank_pool(exp, law_pool, law_obj, law_constraints)
    law_eta, law_rows = _audit_law(
        exp, law_ranked, law_bank, L_max=L_max,
        limit=int(opt.get("law_exact_audit_candidates", 20)),
        min_valid=int(opt.get("law_min_exact_valid", 6)),
        mandatory=[population_eta],
    )
    R_star = float(exp.exact_finite_result(law_eta, law_bank)["value"])
    R_max = R_star + eps_r

    # Shared differentiable law constraints for transport stages.
    fast_R_anchor = float(jax.jit(lambda e: exp.finite_risk(e, law_grad_bank))(law_eta))
    fast_R_max = fast_R_anchor + eps_r
    r_scale = max(eps_r, 1.0e-6 * max(abs(fast_R_anchor), 1.0), 1.0e-10)
    def law_intersection_slack(eta):
        L = exp.population_loss(eta)
        R = exp.finite_risk(eta, law_grad_bank)
        return jnp.maximum((L - fast_L_max) / l_scale, (R - fast_R_max) / r_scale)
    action_constraints = geometry + ((law_intersection_slack, 0.0),)

    # ------------------------------------------------------------------
    # 3/4 tangent action
    # ------------------------------------------------------------------
    grad_tan_bank = _prefix_bank(action_bank, int(opt.get("tangent_gradient_trials", min(4, int(action_bank.sample_indices.shape[0])))))
    local = _local_cloud(
        exp, [law_eta, population_eta],
        count_per_center=int(opt.get("tangent_local_starts", 12)),
        scale=float(opt.get("tangent_local_scale", 0.08)), seed=int(cfg["seed"]) + 401,
    )
    tangent_starts = jnp.stack(_dedupe(exp, [law_eta, population_eta] + list(starts) + local))
    tangent_obj_raw = lambda eta: exp.tangent_action_gradient(eta, grad_tan_bank)
    tan_anchor = max(float(tangent_obj_raw(law_eta)), 1.0e-12)
    tangent_obj = lambda eta: tangent_obj_raw(eta) / tan_anchor
    tan_cfg = _opt_cfg(cfg, "tangent")
    print("[3/4] optimizing tangent correction action", flush=True)
    tan_candidates = optimize_multistart_candidates(
        tangent_obj, tangent_starts, tan_cfg, constraints=action_constraints,
        canonicalize=exp.family.canonicalize,
    
        vectorize_starts=False,) if tan_cfg.steps > 0 else []
    tan_pool = _dedupe(exp, list(tangent_starts) + [r.eta for r in tan_candidates])
    tan_ranked = _rank_pool(exp, tan_pool, tangent_obj_raw, action_constraints)
    tangent_eta, tangent_rows = _audit_action(
        "tangent", exp, tan_ranked, law_bank, action_bank,
        L_max=L_max, R_max=R_max,
        exact_action=lambda eta: exp.exact_tangent_result(eta, action_bank),
        audit_limit=int(opt.get("tangent_exact_audit_candidates", 24)),
        finalist_count=int(opt.get("tangent_exact_rescore_candidates", 8)),
        mandatory=[law_eta, population_eta],
    )

    # ------------------------------------------------------------------
    # 4/4 full weighted-Poisson action
    # ------------------------------------------------------------------
    grad_full_bank = _prefix_bank(action_bank, int(opt.get("full_gradient_trials", min(2, int(action_bank.sample_indices.shape[0])))))
    local_full = _local_cloud(
        exp, [law_eta, tangent_eta, population_eta],
        count_per_center=int(opt.get("full_local_starts", 12)),
        scale=float(opt.get("full_local_scale", 0.06)), seed=int(cfg["seed"]) + 501,
    )
    full_starts = jnp.stack(_dedupe(exp, [law_eta, tangent_eta, population_eta] + list(starts) + local_full))
    full_raw = lambda eta: exp.full_action_gradient(eta, grad_full_bank)
    full_anchor = max(float(full_raw(law_eta)), 1.0e-12)
    full_obj = lambda eta: full_raw(eta) / full_anchor
    full_cfg = _opt_cfg(cfg, "full")
    print("[4/4] optimizing lower-resolution full weighted-Poisson action proxy", flush=True)
    full_candidates = optimize_multistart_candidates(
        full_obj, full_starts, full_cfg, constraints=action_constraints,
        canonicalize=exp.family.canonicalize, vectorize_starts=False,
    ) if full_cfg.steps > 0 else []
    full_pool = _dedupe(exp, list(full_starts) + [r.eta for r in full_candidates])
    full_ranked = _rank_pool(exp, full_pool, full_raw, action_constraints)
    prescreen_trials = int(opt.get("full_prescreen_trials", min(4, int(action_bank.sample_indices.shape[0]))))
    full_eta, full_rows = _audit_action(
        "full", exp, full_ranked, law_bank, action_bank,
        L_max=L_max, R_max=R_max,
        exact_action=lambda eta: exp.exact_full_result(eta, action_bank),
        exact_prescreen=lambda eta: exp.exact_full_result(eta, action_bank, trial_count=prescreen_trials),
        audit_limit=int(opt.get("full_exact_audit_candidates", 30)),
        finalist_count=int(opt.get("full_exact_rescore_candidates", 10)),
        mandatory=[law_eta, tangent_eta, population_eta],
    )

    return {
        "population_eta": population_eta, "law_eta": law_eta,
        "tangent_eta": tangent_eta, "full_eta": full_eta,
        "L_star": L_star, "L_max": L_max, "R_star": R_star, "R_max": R_max,
        "audit": {
            "population": population_rows, "law": law_rows,
            "tangent": tangent_rows, "full": full_rows,
        },
    }
