from __future__ import annotations

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config
from mfsi.design import random_point_sensor_starts
from domain import DoubleGyreTruth
from experiment import (
    VortexExperiment,
    _truth_from_cfg,
    ensure_observation_bank,
    ensure_reference,
    ensure_reference_bank,
    ensure_reference_endpoints,
    ensure_truth_bank,
    prefix_bank,
)


def _check(name, fn, eta):
    value, grad = jax.jit(jax.value_and_grad(fn))(eta)
    v = float(value)
    g = np.asarray(grad, dtype=np.float64)
    if not np.isfinite(v) or not np.all(np.isfinite(g)):
        raise RuntimeError(f"{name}: non-finite value/gradient: {v}, {g}")
    print(f"[gradient-smoke] {name:<16} value={v:.7g} |grad|={np.linalg.norm(g):.3e}", flush=True)
    return {"value": v, "gradient": g.tolist(), "gradient_norm": float(np.linalg.norm(g))}


def _finite_difference_audit(fn, eta, autodiff_gradient, *, step: float = 1.0e-5, rel_tol: float = 5.0e-3):
    """Centered-FD spot check of the implicit/autodiff design gradient.

    The finite-resource risk exercises sensor geometry, cubic reconstruction and the
    empirical I-projection custom VJP.  We deliberately check only two coordinates
    so the engineering smoke remains cheap.
    """
    eval_fn = jax.jit(fn)
    grad = np.asarray(autodiff_gradient, dtype=np.float64)
    indices = sorted({0, int(eta.shape[0]) - 1})
    rows = []
    for idx in indices:
        ep = eta.at[idx].add(step)
        em = eta.at[idx].add(-step)
        fd = float((eval_fn(ep) - eval_fn(em)) / (2.0 * step))
        ad = float(grad[idx])
        scale = max(1.0e-8, abs(fd), abs(ad))
        rel = abs(fd - ad) / scale
        rows.append({"index": int(idx), "autodiff": ad, "finite_difference": fd, "relative_error": rel})
        print(
            f"[gradient-smoke] FD finite R[{idx}] autodiff={ad:.8e} fd={fd:.8e} rel={rel:.3e}",
            flush=True,
        )
        if not np.isfinite(rel) or rel > rel_tol:
            raise RuntimeError(
                f"finite R gradient finite-difference audit failed at coordinate {idx}: "
                f"relative_error={rel:.3e} > {rel_tol:.3e}"
            )
    return {"step": step, "relative_tolerance": rel_tol, "checks": rows}


def main() -> None:
    cfg = load_config(SCRIPT_DIR / "config.json", smoke=True)
    out = SCRIPT_DIR / "outputs" / "gradient_smoke"
    out.mkdir(parents=True, exist_ok=True)
    times = jnp.linspace(0.0, 1.0, int(cfg["poisson"]["time_n"]), dtype=jnp.float64)
    truth = _truth_from_cfg(cfg)
    truth_particles, _ = ensure_truth_bank(truth, cfg, out, times)
    endpoints, endpoint_sig = ensure_reference_endpoints(truth, cfg, out)
    reference, checkpoint, _ = ensure_reference(endpoints, endpoint_sig, cfg, out)
    nodes, velocity, weights = ensure_reference_bank(truth, reference, checkpoint, cfg, out, times)
    exp = VortexExperiment(cfg, reference, truth_particles=truth_particles, reference_nodes=nodes, reference_velocity=velocity, reference_weights=weights)
    bank = ensure_observation_bank(name="gradient", exp=exp, trials=1, namespace=19991, output_dir=out)
    bank = prefix_bank(bank, 1)

    m = cfg["measurement"]
    margin = float(m["boundary_margin"])
    starts = random_point_sensor_starts(
        jax.random.PRNGKey(int(cfg["seed"]) + 9001), 16,
        n_sensors=exp.family.n_sensors,
        x_bounds=(exp.grid.x_min + margin, exp.grid.x_max - margin),
        y_bounds=(exp.grid.y_min + margin, exp.grid.y_max - margin),
        min_sep=float(m["min_sep"]), oversample=256,
    )
    eta = None
    for candidate in starts:
        row = exp._exact_trial_result(candidate, bank, 0, compute_law=False, compute_tangent=False, compute_full=False)
        if row["valid"]:
            eta = candidate
            break
    if eta is None:
        raise RuntimeError("gradient smoke could not find an exact-valid candidate")

    finite_fn = lambda e: exp.finite_risk(e, bank)
    finite = _check("finite R", finite_fn, eta)
    results = {
        "eta": np.asarray(eta).tolist(),
        "centers": np.asarray(exp.family.centers(eta)).tolist(),
        "population": _check("population L", exp.population_loss, eta),
        "finite": finite,
        "finite_gradient_fd_audit": _finite_difference_audit(
            finite_fn, eta, finite["gradient"]
        ),
        "tangent": _check("tangent action", lambda e: exp.tangent_action_gradient(e, bank), eta),
        "full_proxy": _check("full proxy", lambda e: exp.full_action_gradient(e, bank), eta),
    }
    (out / "gradient_smoke.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"results={out / 'gradient_smoke.json'}", flush=True)


if __name__ == "__main__":
    main()
