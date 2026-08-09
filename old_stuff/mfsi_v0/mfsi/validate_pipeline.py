#!/usr/bin/env python3
"""End-to-end oracle validation of the two-component MFSI pipeline.

Run:
    python validate_pipeline.py

This validates the mathematics independently of Tesseract and independently of
any learned density/score model.  The exact 1D weighted-Poisson solution is the
main dynamic oracle; a tiny Deep-Ritz network is only a secondary solver check.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mfsi_components import (
    TARGET,
    calibrate_lambda_unrolled,
    continuity_residual,
    density_time_derivative,
    fourth_moment,
    make_grid,
    mfsi_pipeline,
    phi,
    tangent_only_velocity,
    weighted_l2,
)

jax.config.update("jax_enable_x64", True)

A = 0.8
TIMES = jnp.linspace(0.05, 0.95, 19)
GRID = make_grid(xmax=7.0, n=2001)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def validate_at_time(t: float) -> dict:
    t = jnp.asarray(t)
    ref, fib = mfsi_pipeline(None, GRID, t, A, differentiation="implicit")

    # Independent time derivative of the *projected* density.
    dqdt = density_time_derivative(None, GRID, t, A, differentiation="unrolled")
    res_mfsi = continuity_residual(fib.q, dqdt, fib.velocity, GRID)

    # Moment-tangent ablation: same moment-rate target, wrong full law in general.
    v_tangent = tangent_only_velocity(ref, fib, GRID)
    res_tangent = continuity_residual(fib.q, dqdt, v_tangent, GRID)

    jphi = jnp.stack([jnp.ones_like(GRID.x), 2.0 * GRID.x], axis=-1)
    rate_mfsi = jnp.sum(
        GRID.w[:, None] * fib.q[:, None] * jphi * fib.velocity[:, None], axis=0
    )
    rate_tangent = jnp.sum(
        GRID.w[:, None] * fib.q[:, None] * jphi * v_tangent[:, None], axis=0
    )

    # Independent AD oracle for lambda_dot: differentiate the raw Newton primal.
    ph = phi(GRID.x)
    def lam_of_time(s):
        r, _ = mfsi_pipeline(None, GRID, s, A, differentiation="stop")
        return calibrate_lambda_unrolled(r.log_q_ref, ph, GRID.w, TARGET)
    lambda_dot_ad = jax.jacfwd(lam_of_time)(t)

    return {
        "t": float(t),
        "mean_error": float(abs(fib.moments[0] - TARGET[0])),
        "second_moment_error": float(abs(fib.moments[1] - TARGET[1])),
        "lambda_dot_error": float(jnp.linalg.norm(fib.lambda_dot - lambda_dot_ad)),
        "forcing_mean": float(abs(jnp.sum(GRID.w * fib.q * fib.forcing))),
        "mfsi_continuity_l2": float(
            weighted_l2(res_mfsi / jnp.maximum(fib.q, 1e-8), fib.q, GRID)
        ),
        "tangent_continuity_l2": float(
            weighted_l2(res_tangent / jnp.maximum(fib.q, 1e-8), fib.q, GRID)
        ),
        "mfsi_moment_rate_norm": float(jnp.linalg.norm(rate_mfsi)),
        "tangent_moment_rate_norm": float(jnp.linalg.norm(rate_tangent)),
        "fourth_moment": float(fourth_moment(fib.q, GRID)),
        "projection_distortion": float(fib.projection_distortion),
        "ess_fraction": float(fib.ess_fraction),
        "lambda": np.asarray(fib.lambda_).tolist(),
    }


def train_tiny_ritz(t: float = 0.5, steps: int = 500) -> dict:
    """Small Deep-Ritz check against the exact 1D correction at one time slice."""
    t = jnp.asarray(t)
    rgrid = make_grid(xmax=6.0, n=601)
    _, fib = mfsi_pipeline(None, rgrid, t, A, differentiation="implicit")
    q, h, delta_exact = fib.q, fib.forcing, fib.correction

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    def layer(k, n_in, n_out):
        lim = jnp.sqrt(6.0 / (n_in + n_out))
        return (
            jax.random.uniform(k, (n_in, n_out), minval=-lim, maxval=lim),
            jnp.zeros((n_out,), dtype=jnp.float64),
        )

    W1, b1 = layer(k1, 1, 32)
    W2, b2 = layer(k2, 32, 32)
    W3, b3 = layer(k3, 32, 1)
    params = (W1, b1, W2, b2, W3, b3)

    def psi(p, x):
        W1, b1, W2, b2, W3, b3 = p
        y = jnp.tanh(x[..., None] @ W1 + b1)
        y = jnp.tanh(y @ W2 + b2)
        return (y @ W3 + b3)[..., 0]

    def psi_x(p, x):
        return jax.vmap(jax.grad(lambda xx: psi(p, xx)))(x)

    def loss(p):
        y = psi(p, rgrid.x)
        y = y - jnp.sum(rgrid.w * q * y)
        dy = psi_x(p, rgrid.x)
        return jnp.sum(rgrid.w * q * (0.5 * dy * dy + h * y))

    vg = jax.jit(jax.value_and_grad(loss))
    tree_map = jax.tree_util.tree_map
    m = tree_map(jnp.zeros_like, params)
    v = tree_map(jnp.zeros_like, params)
    history = []
    for i in range(1, steps + 1):
        val, g = vg(params)
        m = tree_map(lambda mm, gg: 0.9 * mm + 0.1 * gg, m, g)
        v = tree_map(lambda vv, gg: 0.999 * vv + 0.001 * gg * gg, v, g)
        mh = tree_map(lambda mm: mm / (1.0 - 0.9**i), m)
        vh = tree_map(lambda vv: vv / (1.0 - 0.999**i), v)
        params = tree_map(
            lambda pp, mm, vv: pp - 2e-3 * mm / (jnp.sqrt(vv) + 1e-8),
            params, mh, vh,
        )
        if i in (1, 100, 400, steps):
            history.append([i, float(val)])

    delta_ritz = -psi_x(params, rgrid.x)
    return {
        "weighted_l2_correction_error": float(weighted_l2(delta_ritz - delta_exact, q, rgrid)),
        "loss_history": history,
        "grid": rgrid,
        "fiber": fib,
        "delta_ritz": delta_ritz,
    }


def main() -> None:
    rows = [validate_at_time(float(t)) for t in TIMES]
    ritz = train_tiny_ritz()

    vp = 1.0 - A * A
    fourth_plus = A**4 + 6.0 * A * A * vp + 3.0 * vp * vp
    summary = {
        "architecture": "ReferenceTransport -> MomentFiberRealizer",
        "differentiation": "implicit VJP through calibration; plain JAX everywhere else",
        "a": A,
        "grid": {"xmax": 7.0, "n": int(GRID.x.size)},
        "max_mean_error": max(r["mean_error"] for r in rows),
        "max_second_moment_error": max(r["second_moment_error"] for r in rows),
        "max_lambda_dot_error": max(r["lambda_dot_error"] for r in rows),
        "max_abs_forcing_mean": max(r["forcing_mean"] for r in rows),
        "median_mfsi_continuity_l2": float(np.median([r["mfsi_continuity_l2"] for r in rows])),
        "median_tangent_continuity_l2": float(np.median([r["tangent_continuity_l2"] for r in rows])),
        "median_mfsi_moment_rate_norm": float(np.median([r["mfsi_moment_rate_norm"] for r in rows])),
        "median_tangent_moment_rate_norm": float(np.median([r["tangent_moment_rate_norm"] for r in rows])),
        "fourth_moment_endpoints": {"Q_minus": 3.0, "Q_plus": fourth_plus},
        "ritz_t0_5_weighted_l2_correction_error": ritz["weighted_l2_correction_error"],
        "ritz_loss_history": ritz["loss_history"],
        "per_time": rows,
    }
    (OUT / "validation_metrics.json").write_text(json.dumps(summary, indent=2))

    ts = np.array([r["t"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].semilogy(ts, np.maximum([r["second_moment_error"] for r in rows], 1e-16))
    axes[0, 0].set_title("Projected second-moment error")
    axes[0, 0].set_xlabel("t")

    axes[0, 1].plot(ts, [r["fourth_moment"] for r in rows])
    axes[0, 1].axhline(3.0, linestyle="--", linewidth=1)
    axes[0, 1].axhline(fourth_plus, linestyle="--", linewidth=1)
    axes[0, 1].set_title("Hidden fourth moment changes")
    axes[0, 1].set_xlabel("t")

    axes[1, 0].semilogy(ts, np.maximum([r["mfsi_continuity_l2"] for r in rows], 1e-16), label="MFSI")
    axes[1, 0].semilogy(ts, np.maximum([r["tangent_continuity_l2"] for r in rows], 1e-16), label="moment-tangent only")
    axes[1, 0].set_title("Full continuity residual")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].legend()

    rg, fib = ritz["grid"], ritz["fiber"]
    mask = np.asarray(fib.q) > 1e-4
    axes[1, 1].plot(np.asarray(rg.x)[mask], np.asarray(fib.velocity - fib.correction)[mask], label="reference u")
    axes[1, 1].plot(np.asarray(rg.x)[mask], np.asarray(fib.velocity)[mask], label="exact MFSI v")
    axes[1, 1].plot(np.asarray(rg.x)[mask], np.asarray(fib.velocity - fib.correction + ritz["delta_ritz"])[mask], linestyle="--", label="Deep-Ritz v")
    axes[1, 1].set_title("Poisson / Deep-Ritz at t=0.5")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(OUT / "validation.png", dpi=160)
    plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k != "per_time"}, indent=2))


if __name__ == "__main__":
    main()
