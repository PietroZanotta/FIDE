from __future__ import annotations

"""Engineering-only gradient smoke for the toy MFSI pipeline.

This script complements ``run.py --smoke``:

* ``run.py --smoke`` checks one exact scientific path end-to-end.
* this script checks that the differentiable objectives and one Adam update execute
  through population law, finite-resource law, tangent action, and the weighted-
  Poisson full-action proxy.

It deliberately does *not* run authoritative stage-1/2 candidate selection, because
smoke-resolution reference quadrature/training is intentionally too crude to be a
meaningful scientific optimization benchmark.
"""

import json
import math
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

from domain import ToyEndpointSource
from experiment import (
    ToyExperiment,
    build_reference_bank,
    ensure_reference,
    make_trial_bank,
)
from mfsi.config import load_config
from mfsi.design import random_projective_starts
from mfsi.selection import build_fast_law_evaluator
from mfsi.exact_feasibility import ExactFeasibilityError


def _one_step(name, objective, eta0, *, learning_rate: float = 1.0e-2):
    # A single first-step Adam update has m_hat=g and v_hat=g^2.  Constructing it
    # directly avoids compiling a second copy of each expensive objective merely to
    # test optimizer plumbing; design.py itself is covered by lightweight unit tests.
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    value, grad = value_and_grad(eta0)
    value_f = float(value)
    grad_np = np.asarray(grad, dtype=np.float64)
    if not np.isfinite(value_f) or not np.all(np.isfinite(grad_np)):
        raise RuntimeError(f"{name}: non-finite value/gradient: value={value_f}, grad={grad_np}")
    eta1 = eta0 - float(learning_rate) * grad / (jnp.abs(grad) + 1.0e-8)
    eta1 = jnp.sort(jnp.mod(eta1, 2.0 * jnp.pi))
    print(
        f"[gradient-smoke] {name}: value={value_f:.8g} "
        f"|grad|={np.linalg.norm(grad_np):.3e}",
        flush=True,
    )
    return {
        "value": value_f,
        "gradient": grad_np.tolist(),
        "gradient_norm": float(np.linalg.norm(grad_np)),
        "one_step_endpoint_rad": np.asarray(eta1, dtype=np.float64).tolist(),
    }


def main() -> None:
    cfg = load_config(SCRIPT_DIR / "config.json", smoke=True)
    cfg = dict(cfg)
    cfg["validity"] = dict(cfg.get("validity", {}))
    cfg["validity"].setdefault("max_population_calibration_resid", 1.0e-5)
    cfg["validity"].setdefault("max_finite_calibration_resid", 1.0e-3)
    cfg["validity"].setdefault("min_ess_fraction", 0.03)
    cfg["validity"].setdefault("min_in_domain_base_mass", 0.995)

    # Force a genuinely differentiable stage-4 proxy even if the smoke overlay has
    # zero optimization steps.  The grid cannot exceed the tiny scientific smoke grid.
    opt = cfg.setdefault("optimization", {})
    opt["full_gradient_trials"] = 1
    opt["full_gradient_time_n"] = min(3, int(cfg["poisson"]["time_n"]))
    opt["full_gradient_grid_n"] = min(11, int(cfg["poisson"]["grid_n"]))
    opt["full_gradient_cg_tol"] = max(float(opt.get("full_gradient_cg_tol", 1.0e-5)), 1.0e-5)
    opt["full_gradient_cg_maxiter"] = min(int(opt.get("full_gradient_cg_maxiter", 80)), 80)

    out = SCRIPT_DIR / "outputs" / "gradient_smoke"
    out.mkdir(parents=True, exist_ok=True)

    endpoints = ToyEndpointSource(
        radius=float(cfg["population"]["radius"]),
        sigma=float(cfg["population"]["sigma"]),
    )
    reference, checkpoint, _ = ensure_reference(endpoints, cfg, out)
    times = jnp.linspace(0.0, 1.0, int(cfg["poisson"]["time_n"]), dtype=jnp.float64)
    nodes, velocity, weights, _ = build_reference_bank(reference, endpoints, times, cfg)
    exp = ToyExperiment(
        cfg,
        reference,
        reference_nodes=nodes,
        reference_velocity=velocity,
        reference_weights=weights,
    )

    bank = make_trial_bank(
        exp.population,
        exp.times,
        exp.acq_idx,
        finite_n=int(cfg["measurement"]["finite_n"]),
        trials=1,
        seed=int(cfg["seed"]),
        namespace=int(cfg.get("randomness", {}).get("selection_namespace", 8890)) + 70001,
    )

    min_sep = math.radians(float(cfg["measurement"]["min_sep_deg"]))
    probes = random_projective_starts(
        jax.random.PRNGKey(int(cfg["seed"]) + 70123),
        64,
        min_sep_rad=min_sep,
    )
    eta0 = None
    for candidate in probes:
        candidate = exp.family.canonicalize(candidate)
        try:
            exp._exact_polytope(candidate)
            row = exp._exact_trial_result(
                candidate,
                bank,
                0,
                compute_law=False,
                compute_tangent=False,
                compute_full=False,
            )
        except ExactFeasibilityError:
            continue
        if bool(row["valid"]):
            eta0 = candidate
            break
    if eta0 is None:
        raise RuntimeError(
            "gradient smoke could not find an exact-valid finite/noisy probe; "
            "run the ordinary smoke and inspect its feasibility diagnostics"
        )

    print(
        "[gradient-smoke] ENGINEERING ONLY: testing differentiable objectives; "
        "authoritative stage-selection audits are intentionally bypassed",
        flush=True,
    )
    print(
        f"[gradient-smoke] probe_deg={np.degrees(np.asarray(eta0)).tolist()} "
        f"proxy_grid={exp.full_gradient_grid.n}x{exp.full_gradient_grid.n} "
        f"proxy_times={len(exp.full_gradient_time_idx)}",
        flush=True,
    )

    # Stages 1/2 actually optimize the accelerated batched law evaluator, so test
    # those exact gradient graphs rather than the slower authoritative rescoring path.
    fast = build_fast_law_evaluator(exp, bank)

    results = {}
    results["population"] = _one_step(
        "population L (fast search graph)",
        fast.population_loss,
        eta0,
    )
    results["finite"] = _one_step(
        "finite R (fast search graph)",
        fast.finite_risk,
        eta0,
    )
    results["tangent"] = _one_step(
        "tangent A_tan",
        lambda eta: exp.tangent_action_gradient(eta, bank),
        eta0,
    )
    # Sequential starts avoid building a vmap of reverse-CG solves; this is also how
    # the optimized full run executes stage 4.
    results["full_proxy"] = _one_step(
        "full A_proxy",
        lambda eta: exp.full_action_gradient(eta, bank),
        eta0,
    )

    payload = {
        "engineering_smoke": True,
        "reference_checkpoint": str(checkpoint),
        "probe_deg": np.degrees(np.asarray(eta0, dtype=np.float64)).tolist(),
        "proxy_grid_n": int(exp.full_gradient_grid.n),
        "proxy_time_n": int(len(exp.full_gradient_time_idx)),
        "results": results,
    }
    (out / "gradient_smoke.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("[gradient-smoke] COMPLETE", flush=True)
    print(f"results={out / 'gradient_smoke.json'}", flush=True)


if __name__ == "__main__":
    main()
