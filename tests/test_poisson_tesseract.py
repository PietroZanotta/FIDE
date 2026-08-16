from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mfsi.linear import implicit_cg
from mfsi.poisson import (
    PoissonConfig,
    solve_weighted_poisson,
    weighted_laplacian,
    weighted_laplacian_diag,
)

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = ROOT / "native" / "poisson_tesseract" / "build"
if str(NATIVE_BUILD) not in sys.path:
    sys.path.insert(0, str(NATIVE_BUILD))

native = pytest.importorskip(
    "_poisson_native",
    reason="build native/poisson_tesseract before running native tests",
)


def _systems(seed: int, shape: tuple[int, int, int]):
    rng = np.random.default_rng(seed)
    q = np.ascontiguousarray(0.5 + rng.random(shape))
    rhs = np.ascontiguousarray(rng.normal(size=shape))
    gauge = np.ascontiguousarray(q / np.linalg.norm(q, axis=(1, 2), keepdims=True))
    return rng, q, rhs, gauge


def _native_solve(q, rhs, gauge, *, dx=0.16, tol=1.0e-10, maxiter=600):
    result = native.solve_batch(q, rhs, gauge, dx, 1.0, tol, maxiter)
    assert np.all(result["converged"]), result
    assert np.all(np.isfinite(result["psi"]))
    return result


def test_native_stencil_and_diagonal_match_jax():
    rng, q, _, gauge = _systems(10, (3, 41, 41))
    psi = np.ascontiguousarray(rng.normal(size=q.shape))
    dx = 0.16

    got_lap = native.weighted_laplacian_batch(psi, q, dx)
    expected_lap = np.stack(
        [np.asarray(weighted_laplacian(jnp.asarray(x), jnp.asarray(c), dx))
         for x, c in zip(psi, q, strict=True)]
    )
    got_diag = native.diagonal_batch(q, gauge, dx, 1.0)
    expected_diag = np.stack(
        [np.asarray(weighted_laplacian_diag(jnp.asarray(c), dx)) + v * v
         for c, v in zip(q, gauge, strict=True)]
    )

    assert np.linalg.norm(got_lap - expected_lap) / np.linalg.norm(expected_lap) < 1.0e-12
    assert np.linalg.norm(got_diag - expected_diag) / np.linalg.norm(expected_diag) < 1.0e-12


def test_native_solve_residual_and_physical_action_match_jax():
    rng = np.random.default_rng(11)
    q = np.ascontiguousarray(0.4 + rng.random((2, 21, 21)))
    h = np.ascontiguousarray(rng.normal(size=q.shape))
    cfg = PoissonConfig(dx=0.24, operator_floor_rel=2.0e-5, cg_tol=1.0e-10, cg_maxiter=600)
    floor = cfg.operator_floor_rel * np.max(q, axis=(1, 2), keepdims=True)
    q_operator = np.ascontiguousarray(q + floor)
    rhs = np.ascontiguousarray(-(q * h))
    gauge = np.ascontiguousarray(q / np.linalg.norm(q, axis=(1, 2), keepdims=True))
    result = _native_solve(
        q_operator, rhs, gauge, dx=cfg.dx, tol=cfg.cg_tol, maxiter=cfg.cg_maxiter
    )
    assert np.max(result["relative_residual"]) <= 1.1 * cfg.cg_tol

    reference = [
        solve_weighted_poisson(jnp.asarray(q[b]), jnp.asarray(h[b]), cfg)
        for b in range(q.shape[0])
    ]
    expected_psi = np.stack([np.asarray(row.potential) for row in reference])
    got_action = np.asarray([
        cfg.cell_area * np.sum(
            result["psi"][b]
            * np.asarray(weighted_laplacian(
                jnp.asarray(result["psi"][b]), jnp.asarray(q[b]), cfg.dx
            ))
        )
        for b in range(q.shape[0])
    ])
    expected_action = np.asarray([float(row.action) for row in reference])
    assert np.linalg.norm(result["psi"] - expected_psi) / np.linalg.norm(expected_psi) < 1.0e-8
    assert np.linalg.norm(got_action - expected_action) / np.linalg.norm(expected_action) < 1.0e-8


@pytest.mark.parametrize("field", ["q_operator", "rhs", "gauge"])
def test_native_vjp_centered_finite_difference(field: str):
    rng, q, rhs, gauge = _systems(12, (2, 8, 7))
    bar = np.ascontiguousarray(rng.normal(size=q.shape))
    dx = 0.19

    psi = _native_solve(q, rhs, gauge, dx=dx)["psi"]
    lambda_ = _native_solve(q, bar, gauge, dx=dx)["psi"]
    gradients = {
        "q_operator": native.weighted_operator_vjp(psi, lambda_, dx),
        "rhs": lambda_,
        "gauge": native.gauge_vjp(psi, lambda_, gauge, 1.0),
    }
    primals = {"q_operator": q, "rhs": rhs, "gauge": gauge}
    direction = rng.normal(size=q.shape)
    direction /= np.linalg.norm(direction)
    exact = float(np.sum(gradients[field] * direction))

    eps = 3.0e-5
    shifted = []
    for sign in (1.0, -1.0):
        args = dict(primals)
        args[field] = np.ascontiguousarray(args[field] + sign * eps * direction)
        shifted.append(
            float(np.sum(_native_solve(
                args["q_operator"], args["rhs"], args["gauge"], dx=dx
            )["psi"] * bar))
        )
    finite_difference = (shifted[0] - shifted[1]) / (2.0 * eps)
    relative = abs(exact - finite_difference) / max(abs(exact), abs(finite_difference), 1.0e-12)
    assert relative < 5.0e-6


def test_tesseract_jit_gradient_matches_jax_implicit_cg():
    pytest.importorskip("tesseract_jax")
    from mfsi.poisson_tesseract import solve_linear_system_batch_tesseract

    rng, q_np, rhs_np, gauge_np = _systems(13, (2, 12, 11))
    bar = jnp.asarray(rng.normal(size=q_np.shape))
    q, rhs, gauge = map(jnp.asarray, (q_np, rhs_np, gauge_np))
    dx, tol, maxiter = 0.2, 1.0e-9, 500
    height, width = q.shape[-2:]

    def one(q_one, rhs_one, gauge_one):
        gauge_flat = gauge_one.reshape(-1)

        def matvec(z_flat):
            z = z_flat.reshape((height, width))
            return (
                weighted_laplacian(z, q_one, dx).reshape(-1)
                + gauge_flat * jnp.dot(gauge_flat, z_flat)
            )

        diag = weighted_laplacian_diag(q_one, dx).reshape(-1) + gauge_flat**2
        return implicit_cg(
            matvec,
            rhs_one.reshape(-1),
            tol=tol,
            maxiter=maxiter,
            preconditioner=lambda r: r / jnp.maximum(diag, 1.0e-10),
        ).reshape((height, width))

    def jax_loss(q_arg, rhs_arg, gauge_arg):
        return jnp.sum(jax.vmap(one)(q_arg, rhs_arg, gauge_arg) * bar)

    def native_loss(q_arg, rhs_arg, gauge_arg):
        psi = solve_linear_system_batch_tesseract(
            q_arg,
            rhs_arg,
            gauge_arg,
            dx=dx,
            gauge_strength=1.0,
            cg_tol=tol,
            cg_maxiter=maxiter,
        )
        return jnp.sum(psi * bar)

    expected = jax.jit(jax.grad(jax_loss, argnums=(0, 1, 2)))(q, rhs, gauge)
    got = jax.jit(jax.grad(native_loss, argnums=(0, 1, 2)))(q, rhs, gauge)
    jax.block_until_ready((expected, got))
    for expected_field, got_field in zip(expected, got, strict=True):
        diff = np.asarray(expected_field - got_field)
        expected_np = np.asarray(expected_field)
        assert np.linalg.norm(diff) / np.linalg.norm(expected_np) < 1.0e-5
        assert np.max(np.abs(diff)) < 1.0e-5
