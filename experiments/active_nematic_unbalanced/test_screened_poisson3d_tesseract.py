from __future__ import annotations

from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
NATIVE_BUILD = (
    HERE.parents[1]
    / "native"
    / "active_nematic_unbalanced_screened_tesseract"
    / "build"
)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NATIVE_BUILD))

native = pytest.importorskip("_active_nematic_unbalanced_screened_native")

from periodic_numerics import PeriodicGrid3D
from screened_poisson3d_tesseract import (
    is_unbalanced_screened_poisson_available,
    solve_unbalanced_screened_poisson3d_batch_tesseract,
)
from unbalanced_correction import (
    UnbalancedCorrectionConfig,
    solve_unbalanced_screened_poisson_batch_jax,
)


def test_native_manufactured_solution_and_pure_reaction() -> None:
    rng = np.random.default_rng(51)
    shape = (3, 9, 8, 7)
    q = np.ascontiguousarray(0.2 + rng.random(shape))
    potential = np.ascontiguousarray(rng.standard_normal(shape))
    parameters = (0.4, 0.5, 0.3, 0.8)
    rhs = native.screened_operator_batch(potential, q, *parameters)
    result = native.solve_batch(q, rhs, *parameters, 1.0e-10, 500, None)
    solved = np.asarray(result["potential"])
    relative = np.sqrt(
        np.sum((solved - potential) ** 2, axis=(1, 2, 3))
        / np.sum(potential**2, axis=(1, 2, 3))
    )
    assert np.all(np.asarray(result["converged"]))
    assert np.max(relative) < 2.0e-8

    constant_h = np.broadcast_to(
        np.asarray([0.7, -0.2, 1.3])[:, None, None, None], shape
    ).copy(order="C")
    reaction = native.solve_batch(
        q, np.ascontiguousarray(q * constant_h), *parameters, 1.0e-11, 500, None
    )
    np.testing.assert_allclose(
        reaction["potential"], parameters[-1] * constant_h, rtol=0.0, atol=5.0e-10
    )


def test_native_operator_derivatives_match_finite_differences() -> None:
    rng = np.random.default_rng(52)
    shape = (2, 7, 6, 5)
    q = np.ascontiguousarray(0.3 + rng.random(shape))
    potential = np.ascontiguousarray(rng.standard_normal(shape))
    adjoint = np.ascontiguousarray(rng.standard_normal(shape))
    q_dot = np.ascontiguousarray(rng.standard_normal(shape))
    rhs_dot = np.ascontiguousarray(rng.standard_normal(shape))
    parameters = (0.5, 0.4, 0.35, 1.2)
    q_bar = native.operator_q_vjp(potential, adjoint, *parameters)
    epsilon = 2.0e-7
    plus = native.screened_operator_batch(
        potential, np.ascontiguousarray(q + epsilon * q_dot), *parameters
    )
    minus = native.screened_operator_batch(
        potential, np.ascontiguousarray(q - epsilon * q_dot), *parameters
    )
    directional = np.sum(adjoint * (plus - minus)) / (2.0 * epsilon)
    # q_bar is the solve VJP contribution -lambda^T (dA/dq) psi.
    np.testing.assert_allclose(-np.sum(q_bar * q_dot), directional, rtol=2.0e-7)

    effective = native.linearized_rhs(
        potential, q_dot, rhs_dot, *parameters
    )
    np.testing.assert_allclose(
        effective, rhs_dot - (plus - minus) / (2.0 * epsilon), rtol=2.0e-7,
        atol=2.0e-7,
    )


def test_native_solution_matches_jax_reference() -> None:
    rng = np.random.default_rng(53)
    grid = PeriodicGrid3D(5.0, (7, 6, 5), polarity_metric_radius=0.8)
    q = jnp.asarray(0.2 + rng.random((2, *grid.shape)))
    h = jnp.asarray(rng.standard_normal((2, *grid.shape)))
    mass = jnp.asarray([3.0, 4.0])
    config = UnbalancedCorrectionConfig(
        reaction_kappa=0.7, cg_tol=1.0e-9, cg_maxiter=500
    )
    reference = solve_unbalanced_screened_poisson_batch_jax(
        q, h, mass=mass, grid=grid, config=config
    )
    q_np = np.ascontiguousarray(q)
    result = native.solve_batch(
        q_np, np.ascontiguousarray(q_np * np.asarray(h)), grid.dx, grid.dy,
        float(grid.dtheta_metric), config.reaction_kappa, config.cg_tol,
        config.cg_maxiter, None,
    )
    np.testing.assert_allclose(
        result["potential"], reference.potential, rtol=2.0e-7, atol=2.0e-8
    )
    assert is_unbalanced_screened_poisson_available()


def test_tesseract_bridge_supports_jitted_reverse_mode() -> None:
    grid = PeriodicGrid3D(4.0, (6, 6, 4))
    q = jnp.full(
        (1, *grid.shape),
        1.0 / (np.prod(grid.shape) * grid.cell_volume),
    )
    wave = jnp.sin(jnp.arange(grid.shape[0]) * 2.0 * jnp.pi / grid.shape[0])
    h = jnp.broadcast_to(wave[None, :, None, None], q.shape)
    config = UnbalancedCorrectionConfig(cg_tol=1.0e-9, cg_maxiter=300)

    def objective(scale):
        result = solve_unbalanced_screened_poisson3d_batch_tesseract(
            q, scale * h, mass=jnp.ones(1), grid=grid, config=config
        )
        return jnp.sum(result.total_action)

    value, gradient = jax.jit(jax.value_and_grad(objective))(1.0)
    np.testing.assert_allclose(gradient, 2.0 * value, rtol=2.0e-8)
