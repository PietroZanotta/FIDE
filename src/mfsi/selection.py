from __future__ import annotations

import math
import copy
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .cache import fingerprint, load_stage_result, save_stage_result
from .design import (
    OptimizeResult,
    OptimizerConfig,
    optimize_multistart_candidates,
    projective_separation_violation,
)
from .law_objectives import FastLawConfig, FastToyLawEvaluator, TrialBank
from .moments import QuadraticBridgeConfig

Array = jax.Array


def _optimizer_cfg(block: dict[str, Any], prefix: str) -> OptimizerConfig:
    return OptimizerConfig(
        steps=int(block.get(f"{prefix}_steps", block.get("steps", 250))),
        learning_rate=float(block.get(f"{prefix}_learning_rate", block.get("learning_rate", 0.02))),
        beta1=float(block.get("beta1", 0.9)),
        beta2=float(block.get("beta2", 0.999)),
        eps=float(block.get("eps", 1e-8)),
        constraint_penalty=float(block.get("constraint_penalty", 1e4)),
        feasibility_tol=float(block.get("feasibility_tol", 1e-6)),
    )


def _canonical_key(family, eta: Array) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(family.canonicalize(eta), dtype=np.float64), 12))


def _dedupe(family, results: list[OptimizeResult]) -> list[OptimizeResult]:
    out: dict[tuple[float, ...], OptimizeResult] = {}
    for result in results:
        key = _canonical_key(family, result.eta)
        if key not in out or result.value < out[key].value:
            out[key] = result
    return list(out.values())


def _choose_stage2_starts(
    fast: FastToyLawEvaluator,
    family,
    starts: Array,
    population_eta: Array,
    population_limit: float,
    *,
    count: int,
) -> Array:
    """Screen every global start once; optimize only the most promising basins.

    This is not a local shortcut: every configured global start is still evaluated
    under the differentiable full-bank L/R search objectives. Expensive gradient trajectories are
    then allocated to the best feasible basins plus the population incumbent.
    """
    starts = jnp.asarray(starts, dtype=jnp.float64)
    values = []
    pair_eval = jax.jit(fast.population_and_finite)
    for i in range(int(starts.shape[0])):
        eta = family.canonicalize(starts[i])
        L, R = pair_eval(eta)
        values.append((float(L), float(R), eta))

    feasible = [x for x in values if x[0] <= population_limit + 1e-12]
    feasible.sort(key=lambda x: x[1])
    chosen = [family.canonicalize(population_eta)]
    for _, _, eta in feasible:
        if len(chosen) >= max(1, count):
            break
        key = _canonical_key(family, eta)
        if not any(_canonical_key(family, x) == key for x in chosen):
            chosen.append(eta)

    # If the strict population screen leaves too few starts, fill by lowest
    # population violation, then finite risk. Exact re-scoring still enforces Lmax.
    if len(chosen) < max(1, count):
        values.sort(key=lambda x: (max(0.0, x[0] - population_limit), x[1]))
        for _, _, eta in values:
            if len(chosen) >= max(1, count):
                break
            key = _canonical_key(family, eta)
            if not any(_canonical_key(family, x) == key for x in chosen):
                chosen.append(eta)

    return jnp.stack(chosen)


def build_fast_law_evaluator(exp, selection_bank: TrialBank) -> FastToyLawEvaluator:
    cfg = exp.cfg
    pop_cfg = cfg["population"]
    validity = cfg.get("validity", {})
    feasibility = cfg.get("feasibility", {})
    raster = cfg.get("raster", {})

    alpha_n = int(pop_cfg.get("alpha_quadrature_n", pop_cfg.get("alpha_n", 5)))
    alphas, alpha_weights = exp.population.alpha_quadrature(alpha_n)
    population_masses = jax.vmap(lambda a: exp.population.masses(exp.times, a))(alphas)
    population_masses = population_masses.reshape(
        (population_masses.shape[0], population_masses.shape[1], -1)
    )

    return FastToyLawEvaluator(
        family=exp.family,
        projector=exp.projector,
        grid=exp.grid,
        times=exp.times,
        time_weights=exp.time_w,
        acq_idx=exp.acq_idx,
        heldout_idx=exp.heldout_idx,
        population_idx=exp.population_idx,
        reference_nodes=exp.reference_nodes,
        reference_base_weights=exp.reference_weights,
        reference_in_domain=exp.reference_in_domain,
        population_masses=population_masses,
        population_alpha_weights=alpha_weights,
        selection_bank=selection_bank,
        mmd_kernel=exp.mmd_kernel,
        support_directions=exp.support_directions,
        moment_cfg=exp.moment_cfg,
        cfg=FastLawConfig(
            finite_n=int(cfg["measurement"]["finite_n"]),
            obs_noise_std=float(cfg["measurement"]["obs_noise_std"]),
            variance_floor=float(exp.moment_cfg.variance_floor),
            raster_bandwidth=float(raster.get("bandwidth", 0.0)),
            raster_truncate=float(raster.get("truncate", 4.0)),
            feasibility_margin=float(exp.feasibility_margin),
            feasibility_tol=float(feasibility.get("feasibility_tol", feasibility.get("tol", 1e-9))),
            max_finite_calibration_resid=float(validity.get("max_finite_calibration_resid", 1e-3)),
            max_population_calibration_resid=float(validity.get("max_population_calibration_resid", 1e-5)),
            min_ess_fraction=float(validity.get("min_ess_fraction", 0.03)),
            min_in_domain_base_mass=float(validity.get("min_in_domain_base_mass", 0.995)),
        ),
    )


def _exact_population(exp, eta: Array) -> dict[str, Any]:
    return exp.exact_population_result(jnp.asarray(eta, dtype=jnp.float64))


def _exact_finite(exp, eta: Array, selection_bank: TrialBank) -> dict[str, Any]:
    return exp.exact_finite_result(jnp.asarray(eta, dtype=jnp.float64), selection_bank)


def _candidate_pool(family, *groups) -> list[Array]:
    out: dict[tuple[float, ...], Array] = {}
    for group in groups:
        if group is None:
            continue
        if isinstance(group, list) and not group:
            continue
        if isinstance(group, list) and isinstance(group[0], OptimizeResult):
            seq = [r.eta for r in group]
        else:
            arr = jnp.asarray(group, dtype=jnp.float64)
            seq = [arr] if arr.ndim == 1 else [arr[i] for i in range(int(arr.shape[0]))]
        for eta in seq:
            eta = family.canonicalize(eta)
            out[_canonical_key(family, eta)] = eta
    return list(out.values())


def optimize_population_and_law(
    *,
    exp,
    selection_bank: TrialBank,
    dense_selection_bank: TrialBank,
    starts: Array,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Optimized stages 1-2 with authoritative final re-scoring.

    Stage 2 differs from the earlier implementation in two efficiency-only ways:

    1. L(eta) and R(eta) share one sensor-geometry graph inside the population
       penalty, instead of recomputing Phi_eta twice per Adam step.
    2. all configured global starts are screened once, then only the best few
       basins receive full gradient trajectories.

    Final L*/R* and selected designs still come from the authoritative experiment
    evaluator, not from the optimization helper.
    """
    cfg = exp.cfg
    opt = cfg["optimization"]
    law_cfg = cfg["law"]
    if "epsilon_l" not in law_cfg or "epsilon_r" not in law_cfg:
        raise KeyError(
            "law config must define additive epsilon_l and epsilon_r; "
            "multiplicative tau_l/tau_r are no longer used"
        )
    output_dir = Path(output_dir)
    fast = build_fast_law_evaluator(exp, dense_selection_bank)
    full_trial_count = int(dense_selection_bank.masses.shape[0])
    law_gradient_trials = int(opt.get("law_gradient_trials", min(full_trial_count, 12)))
    law_gradient_trials = max(1, min(law_gradient_trials, full_trial_count))
    if law_gradient_trials < full_trial_count:
        gradient_bank = TrialBank(
            masses=dense_selection_bank.masses[:law_gradient_trials],
            sample_indices=dense_selection_bank.sample_indices[:law_gradient_trials],
            detector_z=dense_selection_bank.detector_z[:law_gradient_trials],
            alphas=dense_selection_bank.alphas[:law_gradient_trials],
        )
        gradient_fast = build_fast_law_evaluator(exp, gradient_bank)
    else:
        gradient_fast = fast

    fast_population_eval = jax.jit(fast.population_loss)
    fast_finite_eval = jax.jit(fast.finite_risk)

    min_sep = math.radians(float(cfg["measurement"]["min_sep_deg"]))
    sep_constraint = (projective_separation_violation(min_sep), 0.0)

    # Stage 1/2 do not depend on epsilon_r: epsilon_r is only the downstream
    # admissible risk slack used by tangent/full selection.  Excluding it from
    # the stage-1/2 cache signature makes Pareto sweeps over epsilon_r reuse the
    # expensive population/Law solution safely.  R_max is recomputed from the
    # cached R_star and the *current* epsilon_r below.
    stage12_cfg = copy.deepcopy(cfg)
    stage12_cfg.setdefault("law", {})["epsilon_r"] = "<downstream-only>"
    # Tangent/full optimizer fidelity and action/validation Monte Carlo counts are
    # downstream of the population/Law solution and must not invalidate stage 1/2
    # during Pareto or proxy-convergence sweeps.
    stage12_opt = stage12_cfg.setdefault("optimization", {})
    for key in list(stage12_opt):
        if key.startswith("tangent_") or key.startswith("full_"):
            stage12_opt.pop(key, None)
    stage12_rnd = stage12_cfg.setdefault("randomness", {})
    for key in ("action_trials", "validation_trials", "validation_namespace", "bootstrap_reps"):
        stage12_rnd.pop(key, None)
    base_signature = fingerprint({
        "config": stage12_cfg,
        "reference_shape": tuple(np.asarray(exp.reference_nodes).shape),
        "selection_shape": tuple(np.asarray(dense_selection_bank.masses).shape),
        "law_evaluator_revision": 6,
    })

    # ------------------------------------------------------------------
    # 1/4 population law
    # ------------------------------------------------------------------
    pop_sig = fingerprint({"base": base_signature, "stage": "population_v4_additive_exact_audit"})
    cached = load_stage_result(output_dir, "population_selection", signature=pop_sig)
    if cached is None:
        print("[1/4] optimizing population-law objective (batched exact-law path)", flush=True)
        pop_candidates = optimize_multistart_candidates(
            fast.population_loss,
            starts,
            _optimizer_cfg(opt, "population"),
            constraints=(sep_constraint,),
            canonicalize=exp.family.canonicalize,
        )
        pop_rows = []
        # Authoritative audit of every global seed and optimizer endpoint.
        for eta in _candidate_pool(exp.family, starts, pop_candidates):
            if float(sep_constraint[0](eta)) > 0.0:
                continue
            exact = _exact_population(exp, eta)
            pop_rows.append({
                "eta": eta,
                "fast": float(fast_population_eval(eta)),
                "exact": float(exact["value"]),
                "valid": bool(exact["valid"]),
                "diagnostics": {k: v for k, v in exact.items() if k not in ("value", "valid")},
            })
        valid_pop = [r for r in pop_rows if r["valid"] and np.isfinite(r["exact"])]
        if not valid_pop:
            raise RuntimeError("No scientifically valid population candidate after exact audit")
        best = min(valid_pop, key=lambda r: r["exact"])
        L_star = float(best["exact"])
        cached = {
            "eta": np.asarray(best["eta"]).tolist(),
            "L_star": L_star,
            "L_max": L_star + float(law_cfg["epsilon_l"]),
            "audited_candidate_count": len(pop_rows),
            "valid_candidate_count": len(valid_pop),
            "rescored": [
                {
                    "eta": np.asarray(r["eta"]).tolist(),
                    "fast": r["fast"],
                    "exact": r["exact"] if np.isfinite(r["exact"]) else None,
                    "valid": r["valid"],
                    "diagnostics": r["diagnostics"],
                }
                for r in pop_rows
            ],
        }
        save_stage_result(output_dir, "population_selection", signature=pop_sig, result=cached)
    else:
        print("[1/4] reusing cached population-law optimum", flush=True)

    population_eta = jnp.asarray(cached["eta"], dtype=jnp.float64)
    L_star = float(cached["L_star"])
    L_max = float(cached["L_max"])

    # ------------------------------------------------------------------
    # 2/4 finite-resource law risk
    # ------------------------------------------------------------------
    law_sig = fingerprint({"base": base_signature, "stage": "finite_law_v5_proxy_ranked_exact_audit", "L_max": L_max})
    law_cached = load_stage_result(output_dir, "finite_law_selection", signature=law_sig)
    if law_cached is None:
        fast_L_anchor = float(fast_population_eval(population_eta))
        fast_L_max = fast_L_anchor + float(law_cfg["epsilon_l"])
        stage2_start_count = int(opt.get("law_start_count", min(5, int(starts.shape[0]) + 1)))
        law_starts = _choose_stage2_starts(
            fast,
            exp.family,
            starts,
            population_eta,
            fast_L_max,
            count=stage2_start_count,
        )
        print(
            "[2/4] optimizing finite-resource law risk "
            f"(full-bank screening={full_trial_count} trials; gradient bank={law_gradient_trials}; "
            f"all {int(starts.shape[0])} starts screened; {int(law_starts.shape[0])} basins optimized)",
            flush=True,
        )

        opt_cfg = _optimizer_cfg(opt, "law")
        grad_fast_L_anchor = float(jax.jit(gradient_fast.population_loss)(population_eta))
        grad_fast_L_max = grad_fast_L_anchor + float(law_cfg["epsilon_l"])
        stage2_objective = lambda eta: gradient_fast.finite_penalized_by_population(
            eta,
            population_limit=grad_fast_L_max,
            penalty=opt_cfg.constraint_penalty,
        )
        # Population feasibility is already inside the shared L/R graph.  Keep
        # only the cheap projective-separation constraint in the generic optimizer.
        law_candidates = optimize_multistart_candidates(
            stage2_objective,
            law_starts,
            opt_cfg,
            constraints=(sep_constraint,),
            canonicalize=exp.family.canonicalize,
        )
        law_candidates = _dedupe(exp.family, law_candidates)
        law_candidates.sort(key=lambda r: float(fast_finite_eval(r.eta)))

        # Exact finite-law scoring over the full CRN bank is one of the dominant
        # stage-2 costs.  Rank the complete seed/endpoint pool with the differentiable
        # full-bank surrogate, then spend authoritative ConvexHull/hard-I-projection
        # audits on the strongest candidates.  The population incumbent is mandatory.
        # If the initial shortlist yields no valid design, progressively audit the
        # remainder rather than failing because of surrogate mismatch.
        candidate_pool = list(_candidate_pool(exp.family, starts, population_eta, law_candidates))
        ranked = []
        pop_key = _canonical_key(exp.family, population_eta)
        for eta in candidate_pool:
            if float(sep_constraint[0](eta)) > 0.0:
                continue
            fast_L = float(fast_population_eval(eta))
            fast_R = float(fast_finite_eval(eta))
            ranked.append((
                0 if _canonical_key(exp.family, eta) == pop_key else 1,
                max(0.0, fast_L - fast_L_max),
                fast_R,
                eta,
            ))
        ranked.sort(key=lambda x: (x[0], x[1], x[2]))
        audit_limit = int(opt.get("law_exact_audit_candidates", min(6, len(ranked))))
        audit_limit = max(1, min(audit_limit, len(ranked))) if ranked else 0

        exact_rows = []
        def audit_one(eta):
            pop = _exact_population(exp, eta)
            if not pop["valid"] or float(pop["value"]) > L_max + 1e-12:
                return {
                    "eta": eta,
                    "fast_R": float(fast_finite_eval(eta)),
                    "exact_L": float(pop["value"]),
                    "exact_R": float("inf"),
                    "valid": False,
                    "reason": "population_invalid_or_outside_Lmax",
                }
            fin = _exact_finite(exp, eta, selection_bank)
            return {
                "eta": eta,
                "fast_R": float(fast_finite_eval(eta)),
                "exact_L": float(pop["value"]),
                "exact_R": float(fin["value"]),
                "valid": bool(fin["valid"]),
                "reason": "ok" if fin["valid"] else "one_or_more_finite_trials_invalid",
            }

        for _, _, _, eta in ranked[:audit_limit]:
            exact_rows.append(audit_one(eta))

        # Robust fallback: expand the exact audit only when the proxy shortlist was
        # unlucky near a feasibility boundary.
        if not any(r["valid"] and np.isfinite(r["exact_R"]) for r in exact_rows):
            for _, _, _, eta in ranked[audit_limit:]:
                exact_rows.append(audit_one(eta))
                if exact_rows[-1]["valid"] and np.isfinite(exact_rows[-1]["exact_R"]):
                    break

        valid_rows = [
            r for r in exact_rows
            if r["valid"] and r["exact_L"] <= L_max + 1e-12 and np.isfinite(r["exact_R"])
        ]
        if not valid_rows:
            raise RuntimeError("No scientifically valid population-feasible finite-law candidate")
        best = min(valid_rows, key=lambda r: r["exact_R"])
        R_star = float(best["exact_R"])
        law_cached = {
            "eta": np.asarray(best["eta"]).tolist(),
            "R_star": R_star,
            "R_max": R_star + float(law_cfg["epsilon_r"]),
            "audited_candidate_count": len(exact_rows),
            "valid_candidate_count": len(valid_rows),
            "rescored": [
                {
                    "eta": np.asarray(r["eta"]).tolist(),
                    "fast_R": r["fast_R"],
                    "exact_L": r["exact_L"] if np.isfinite(r["exact_L"]) else None,
                    "exact_R": r["exact_R"] if np.isfinite(r["exact_R"]) else None,
                    "valid": r["valid"],
                    "reason": r["reason"],
                }
                for r in exact_rows
            ],
        }
        save_stage_result(output_dir, "finite_law_selection", signature=law_sig, result=law_cached)
    else:
        print("[2/4] reusing cached finite-resource law optimum", flush=True)

    return {
        "population_eta": population_eta,
        "law_eta": jnp.asarray(law_cached["eta"], dtype=jnp.float64),
        "L_star": L_star,
        "L_max": L_max,
        "R_star": float(law_cached["R_star"]),
        "R_max": float(law_cached["R_star"]) + float(law_cfg["epsilon_r"]),
        "fast_evaluator": fast,
        "gradient_evaluator": gradient_fast,
        "law_gradient_trials": law_gradient_trials,
    }
