"""Fast percentage-budget design selection for the unbalanced experiment.

The expensive scientific evaluators remain unchanged.  Optimization uses one
bounded Adam search per objective, while every reported design is selected by
the complete declared selection bank.  The Full stage uses a small exact
pre-screen before its complete-bank action evaluation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.design import OptimizeResult, OptimizerConfig, optimize_multistart_candidates

try:
    from .measurements import periodic_separation_violation, random_periodic_sensor_starts
except ImportError:  # pragma: no cover - direct script execution
    from measurements import periodic_separation_violation, random_periodic_sensor_starts


Array = jax.Array


def percentage_risk_budget(risk_star: float, relative: float) -> tuple[float, float]:
    """Return the additive allowance and ceiling for a relative risk budget."""
    risk_star = float(risk_star)
    relative = float(relative)
    if not np.isfinite(risk_star):
        raise ValueError("risk_star must be finite")
    if not np.isfinite(relative) or relative < 0.0:
        raise ValueError("relative risk allowance must be finite and nonnegative")
    epsilon_r = relative * abs(risk_star)
    return epsilon_r, risk_star + epsilon_r


def _prefix_bank(bank, count: int):
    count = max(1, min(int(count), int(bank.plus_sample_indices.shape[0])))
    return type(bank)(*(value[:count] for value in bank))


def _optimizer_config(opt: dict[str, Any], stage: str) -> OptimizerConfig:
    return OptimizerConfig(
        steps=int(opt.get(f"{stage}_steps", 20)),
        learning_rate=float(opt.get(f"{stage}_learning_rate", 0.006)),
        constraint_penalty=float(opt.get("constraint_penalty", 1.0e4)),
        feasibility_tol=float(opt.get("feasibility_tol", 1.0e-6)),
    )


def _dedupe(exp, candidates: list[OptimizeResult]) -> list[OptimizeResult]:
    unique: dict[tuple[float, ...], OptimizeResult] = {}
    for candidate in candidates:
        eta = exp.sensors.canonicalize(candidate.eta)
        key = tuple(np.round(np.asarray(eta), 10))
        previous = unique.get(key)
        normalized = OptimizeResult(
            eta=eta,
            value=float(candidate.value),
            feasible=bool(candidate.feasible),
            violations=tuple(float(v) for v in candidate.violations),
        )
        if previous is None or normalized.value < previous.value:
            unique[key] = normalized
    return list(unique.values())


def _explicit_candidate(exp, eta: Array, value: float = float("inf")) -> OptimizeResult:
    """Materialize a frozen geometry as a candidate without optimizing it away."""
    canonical = exp.sensors.canonicalize(jnp.asarray(eta, dtype=jnp.float64))
    return OptimizeResult(
        eta=canonical,
        value=float(value),
        feasible=True,
        violations=(),
    )


def _configured_candidates(exp, key: str) -> list[OptimizeResult]:
    values = exp.cfg.get("optimization", {}).get(key, [])
    return [_explicit_candidate(exp, eta) for eta in values]


def _progress(stage: str):
    return lambda completed, total: print(
        f"percentage design stage={stage} optimized={completed}/{total}",
        flush=True,
    )


def _local_cloud(exp, centers, *, count: int, scale: float, seed: int) -> list[Array]:
    """Deterministic feasible perturbations around certified incumbents."""
    count = max(0, int(count))
    if count == 0:
        return []
    min_sep = float(exp.cfg["measurement"].get("min_sep", 0.0))
    violation = periodic_separation_violation(
        min_sep,
        n_sensors=exp.sensors.n_sensors,
        box_size=exp.sensors.box_size,
    )
    rng = np.random.default_rng(int(seed))
    centers = [np.asarray(center, dtype=np.float64) for center in centers]
    out: list[Array] = []
    attempts = max(32, 16 * count)
    for index in range(attempts):
        center = centers[index % len(centers)]
        eta = exp.sensors.canonicalize(
            jnp.asarray(center + rng.normal(scale=float(scale), size=center.shape))
        )
        if float(violation(eta)) <= 0.0:
            out.append(eta)
            if len(out) == count:
                return out
    raise RuntimeError(
        f"could not generate {count} feasible local starts in {attempts} attempts"
    )


def _normalized_risk_constraint(exp, bank, anchor_eta: Array, relative: float):
    """Return a dimensionless proxy-risk constraint and its diagnostics."""
    if hasattr(exp, "mean_metric_by_view"):
        risk = lambda eta: exp.mean_metric_by_view(eta, bank, "law_risk")
        anchor_by_view = jnp.asarray(risk(anchor_eta), dtype=jnp.float64)
        maxima = anchor_by_view + float(relative) * jnp.abs(anchor_by_view)
        scales = jnp.maximum(
            float(relative) * jnp.abs(anchor_by_view),
            jnp.maximum(1.0e-6 * jnp.maximum(jnp.abs(anchor_by_view), 1.0), 1.0e-10),
        )
        slack = lambda eta: jnp.max((risk(eta) - maxima) / scales)
        return (
            (slack, 0.0),
            float(jnp.max(anchor_by_view)),
            float(jnp.max(maxima)),
            float(jnp.max(scales)),
        )
    risk = lambda eta: exp.mean_metric(eta, bank, "law_risk")
    anchor = float(risk(anchor_eta))
    epsilon, maximum = percentage_risk_budget(anchor, relative)
    scale = max(epsilon, 1.0e-6 * max(abs(anchor), 1.0), 1.0e-10)
    slack = lambda eta: (risk(eta) - maximum) / scale
    return (slack, 0.0), anchor, maximum, scale


def _audit_law(exp, bank, candidates, limit: int):
    ordered = sorted(_dedupe(exp, candidates), key=lambda row: row.value)
    rows = []
    for candidate in ordered[: max(1, int(limit))]:
        if not candidate.feasible:
            continue
        exact = exp.audit_metric(candidate.eta, bank, "law_risk")
        rows.append((candidate, exact, exact))
    valid = [row for row in rows if row[1]["valid"] and np.isfinite(row[1]["value"])]
    if not valid:
        raise RuntimeError("no exact-valid percentage Law candidate")
    return min(valid, key=lambda row: row[1]["value"]), rows


def _risk_passes(risk: dict[str, Any], risk_max: float, risk_view_maxima) -> bool:
    if risk_view_maxima is None or "view_values" not in risk:
        return float(risk["value"]) <= float(risk_max) + 1.0e-12
    values = np.asarray(risk["view_values"], dtype=np.float64)
    maxima = np.asarray(risk_view_maxima, dtype=np.float64)
    return bool(values.shape == maxima.shape and np.all(values <= maxima + 1.0e-12))


def _law_screen(
    exp, bank, candidates, *, risk_max: float, risk_view_maxima, limit: int, mandatory
):
    mandatory_keys = {
        tuple(np.round(np.asarray(exp.sensors.canonicalize(eta)), 10))
        for eta in mandatory
    }
    ordered = sorted(_dedupe(exp, candidates), key=lambda row: row.value)
    first = [
        row for row in ordered
        if tuple(np.round(np.asarray(row.eta), 10)) in mandatory_keys
    ]
    rest = [
        row for row in ordered
        if tuple(np.round(np.asarray(row.eta), 10)) not in mandatory_keys
    ]
    screened = []
    for candidate in (first + rest)[: max(len(first), int(limit))]:
        if not candidate.feasible:
            continue
        risk = exp.audit_metric(candidate.eta, bank, "law_risk")
        if risk["valid"] and _risk_passes(risk, risk_max, risk_view_maxima):
            screened.append((candidate, risk))
    if not screened:
        raise RuntimeError("no exact law-feasible percentage action candidate")
    return screened


def _audit_tangent(exp, bank, screened):
    rows = []
    for candidate, risk in screened:
        action = exp.audit_metric(candidate.eta, bank, "tangent_action")
        rows.append((candidate, action, risk))
    valid = [row for row in rows if row[1]["valid"] and np.isfinite(row[1]["value"])]
    if not valid:
        raise RuntimeError("no exact-valid percentage tangent candidate")
    return min(valid, key=lambda row: row[1]["value"]), rows


def _audit_full(
    exp,
    bank,
    screened,
    *,
    prescreen_trials: int,
    finalists: int,
    mandatory,
):
    small_bank = _prefix_bank(bank, prescreen_trials)
    prescreened = []
    for candidate, risk in screened:
        action = exp.audit_metric(candidate.eta, small_bank, "full_action")
        if action["valid"] and np.isfinite(action["value"]):
            prescreened.append((float(action["value"]), candidate, risk, action))
    if not prescreened:
        raise RuntimeError("no valid percentage Full candidate survived pre-screen")
    prescreened.sort(key=lambda row: row[0])
    mandatory_keys = {
        tuple(np.round(np.asarray(exp.sensors.canonicalize(eta)), 10))
        for eta in mandatory
    }
    selected = prescreened[: max(1, int(finalists))]
    selected_keys = {
        tuple(np.round(np.asarray(row[1].eta), 10)) for row in selected
    }
    for row in prescreened:
        key = tuple(np.round(np.asarray(row[1].eta), 10))
        if key in mandatory_keys and key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)
    rows = []
    for _, candidate, risk, prescreen in selected:
        action = exp.audit_metric(candidate.eta, bank, "full_action")
        action = dict(action)
        action["prescreen_value"] = float(prescreen["value"])
        rows.append((candidate, action, risk))
    valid = [row for row in rows if row[1]["valid"] and np.isfinite(row[1]["value"])]
    if not valid:
        raise RuntimeError("no exact-valid percentage Full finalist")
    return min(valid, key=lambda row: row[1]["value"]), rows


def _serial(rows):
    return [
        {
            "eta": np.asarray(candidate.eta).tolist(),
            "proxy_value": float(candidate.value),
            "audit": audit,
            "law_screen": risk,
        }
        for candidate, audit, risk in rows
    ]


def optimize_percentage_designs(exp, bank):
    """Select Law, tangent, and Full designs with a relative risk ceiling."""
    opt = exp.cfg["optimization"]
    relative = float(exp.cfg["law"]["max_relative_risk_violation"])
    if not np.isfinite(relative) or relative < 0.0:
        raise ValueError("law.max_relative_risk_violation must be finite and nonnegative")

    available = int(bank.plus_sample_indices.shape[0])
    law_bank = _prefix_bank(bank, int(opt.get("law_gradient_trials", min(8, available))))
    tangent_bank = _prefix_bank(bank, int(opt.get("tangent_gradient_trials", min(8, available))))
    full_bank = _prefix_bank(bank, int(opt.get("full_gradient_trials", min(2, available))))
    starts = random_periodic_sensor_starts(
        jax.random.PRNGKey(int(exp.cfg["seed"]) + 17),
        int(opt.get("start_count", 12)),
        n_sensors=exp.sensors.n_sensors,
        box_size=exp.sensors.box_size,
        min_separation=float(exp.cfg["measurement"].get("min_sep", 0.0)),
        oversample=int(opt.get("start_oversample", 64)),
    )
    geometry = ((periodic_separation_violation(
        float(exp.cfg["measurement"].get("min_sep", 0.0)),
        n_sensors=exp.sensors.n_sensors,
        box_size=exp.sensors.box_size,
    ), 0.0),)

    fixed_law = opt.get("fixed_law_anchor")
    law_raw = lambda eta: exp.mean_metric(eta, law_bank, "law_risk")
    if fixed_law is None:
        print(
            f"[1/3] percentage Law search: {law_bank.plus_sample_indices.shape[0]} "
            "gradient trials, complete-bank exact audit",
            flush=True,
        )
        law_scale = max(float(law_raw(starts[0])), 1.0e-12)
        law_candidates = optimize_multistart_candidates(
            lambda eta: law_raw(eta) / law_scale,
            starts[: int(opt.get("law_start_count", len(starts)))],
            _optimizer_config(opt, "law"),
            constraints=geometry,
            canonicalize=exp.sensors.canonicalize,
            vectorize_starts=False,
            progress_callback=_progress("law"),
        )
    else:
        law_eta = fixed_law["eta"] if isinstance(fixed_law, dict) else fixed_law
        law_candidates = [_explicit_candidate(exp, law_eta, float(law_raw(law_eta)))]
        print("[1/3] percentage Law search: using frozen exact anchor", flush=True)
    law, law_rows = _audit_law(
        exp, bank, law_candidates,
        int(opt.get("law_exact_audit_candidates", 6)),
    )
    law_candidate, law_audit, _ = law
    risk_star = float(law_audit["value"])
    epsilon_r, risk_max = percentage_risk_budget(risk_star, relative)
    risk_view_star = law_audit.get("view_values")
    risk_view_maxima = (
        [float(value) + relative * abs(float(value)) for value in risk_view_star]
        if risk_view_star is not None
        else None
    )
    print(
        f"[budget] R*={risk_star:.8g}; relative allowance={100.0 * relative:.3g}%; "
        f"epsilon_R={epsilon_r:.8g}; R_max={risk_max:.8g}",
        flush=True,
    )

    tangent_risk_constraint, proxy_risk_anchor, proxy_risk_max, _ = (
        _normalized_risk_constraint(exp, tangent_bank, law_candidate.eta, relative)
    )
    tangent_constraints = geometry + (tangent_risk_constraint,)
    tangent_local = _local_cloud(
        exp,
        [law_candidate.eta],
        count=int(opt.get("tangent_local_starts", 0)),
        scale=float(opt.get("tangent_local_scale", 1.0)),
        seed=int(exp.cfg["seed"]) + 401,
    )
    tangent_starts = jnp.stack([
        law_candidate.eta,
        *tangent_local,
        *list(starts[: int(opt.get("tangent_start_count", len(starts)))]),
    ])
    tangent_raw = lambda eta: exp.mean_metric(eta, tangent_bank, "tangent_action")
    tangent_scale = max(float(tangent_raw(law_candidate.eta)), 1.0e-12)
    print(
        f"[2/3] tangent search: {tangent_bank.plus_sample_indices.shape[0]} "
        f"gradient trials, {len(tangent_starts)} starts, normalized risk ceiling "
        f"{proxy_risk_max:.8g}, complete-bank certification",
        flush=True,
    )
    tangent_candidates = optimize_multistart_candidates(
        lambda eta: tangent_raw(eta) / tangent_scale,
        tangent_starts,
        _optimizer_config(opt, "tangent"),
        constraints=tangent_constraints,
        canonicalize=exp.sensors.canonicalize,
        vectorize_starts=False,
        progress_callback=_progress("tangent"),
    )
    # Optimizer outputs are endpoints, not their initial seeds.  Reinsert every
    # incumbent explicitly so the subsequent exact audit cannot optimize it away.
    tangent_candidates = [
        *tangent_candidates,
        _explicit_candidate(exp, law_candidate.eta, float(tangent_raw(law_candidate.eta))),
        *_configured_candidates(exp, "tangent_audit_seed_etas"),
    ]
    tangent_screened = _law_screen(
        exp, bank, tangent_candidates,
        risk_max=risk_max,
        risk_view_maxima=risk_view_maxima,
        limit=int(opt.get("tangent_exact_audit_candidates", 6)),
        mandatory=[law_candidate.eta],
    )
    tangent, tangent_rows = _audit_tangent(exp, bank, tangent_screened)
    tangent_candidate = tangent[0]

    proxy_cfg = dict(exp.cfg)
    proxy_cfg["full_action"] = dict(exp.cfg["full_action"])
    proxy_cfg["full_action"]["grid_shape"] = list(
        opt.get("full_gradient_grid_shape", [20, 20, 10])
    )
    proxy_cfg["full_action"]["cg_tol"] = float(opt.get("full_gradient_cg_tol", 1.0e-6))
    proxy_cfg["full_action"]["cg_maxiter"] = int(opt.get("full_gradient_cg_maxiter", 300))
    if hasattr(exp, "with_config"):
        proxy_exp = exp.with_config(proxy_cfg)
    else:
        proxy_exp = type(exp)(
            proxy_cfg,
            times=exp.times,
            plus=exp.data["plus"],
            minus=exp.data["minus"],
        )
    full_risk_constraint, full_proxy_anchor, full_proxy_max, _ = (
        _normalized_risk_constraint(exp, full_bank, law_candidate.eta, relative)
    )
    full_constraints = geometry + (full_risk_constraint,)
    full_local = _local_cloud(
        exp,
        [law_candidate.eta, tangent_candidate.eta],
        count=int(opt.get("full_local_starts", 0)),
        scale=float(opt.get("full_local_scale", 1.0)),
        seed=int(exp.cfg["seed"]) + 501,
    )
    full_starts = jnp.stack([
        law_candidate.eta,
        tangent_candidate.eta,
        *full_local,
        *list(starts[: int(opt.get("full_start_count", len(starts)))]),
    ])
    full_raw = lambda eta: proxy_exp.mean_metric(eta, full_bank, "full_action")
    full_scale = max(float(full_raw(law_candidate.eta)), 1.0e-12)
    print(
        f"[3/3] Full search: {full_bank.plus_sample_indices.shape[0]} gradient "
        f"trials, {len(full_starts)} starts on "
        f"{proxy_cfg['full_action']['grid_shape']}, normalized risk ceiling "
        f"{full_proxy_max:.8g}, then exact pre-screen",
        flush=True,
    )
    full_candidates = optimize_multistart_candidates(
        lambda eta: full_raw(eta) / full_scale,
        full_starts,
        _optimizer_config(opt, "full"),
        constraints=full_constraints,
        canonicalize=exp.sensors.canonicalize,
        vectorize_starts=False,
        progress_callback=_progress("full"),
    )
    full_candidates = [
        *full_candidates,
        _explicit_candidate(exp, law_candidate.eta, float(full_raw(law_candidate.eta))),
        _explicit_candidate(exp, tangent_candidate.eta, float(full_raw(tangent_candidate.eta))),
        *_configured_candidates(exp, "full_audit_seed_etas"),
    ]
    pareto_incumbent = opt.get("pareto_incumbent_full_eta")
    mandatory_full = [law_candidate.eta, tangent_candidate.eta]
    if pareto_incumbent is not None:
        incumbent = _explicit_candidate(exp, pareto_incumbent, float(full_raw(pareto_incumbent)))
        full_candidates.append(incumbent)
        mandatory_full.append(incumbent.eta)
    full_screened = _law_screen(
        exp, bank, full_candidates,
        risk_max=risk_max,
        risk_view_maxima=risk_view_maxima,
        limit=int(opt.get("full_exact_audit_candidates", 6)),
        mandatory=mandatory_full,
    )
    full, full_rows = _audit_full(
        exp, bank, full_screened,
        prescreen_trials=int(opt.get("full_prescreen_trials", min(4, available))),
        finalists=int(opt.get("full_exact_rescore_candidates", 2)),
        mandatory=mandatory_full,
    )

    selected_full = full
    selected_tangent = tangent
    certified = bool(
        law_audit["valid"]
        and selected_tangent[1]["valid"]
        and selected_tangent[2]["valid"]
        and _risk_passes(selected_tangent[2], risk_max, risk_view_maxima)
        and selected_full[1]["valid"]
        and selected_full[2]["valid"]
        and _risk_passes(selected_full[2], risk_max, risk_view_maxima)
    )

    return {
        "law_eta": law_candidate.eta,
        "tangent_eta": tangent_candidate.eta,
        "full_eta": full[0].eta,
        "risk_star": risk_star,
        "risk_max": risk_max,
        "risk_view_star": risk_view_star,
        "risk_view_maxima": risk_view_maxima,
        "relative_risk_allowance": relative,
        "certified": certified,
        "candidates": {
            "law": _serial(law_rows),
            "tangent": _serial(tangent_rows),
            "full": _serial(full_rows),
        },
    }
