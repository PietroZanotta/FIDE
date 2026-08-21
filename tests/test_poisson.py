import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mfsi.poisson import (
    PoissonConfig,
    solve_weighted_poisson,
    solve_weighted_poisson_physical_direct_batch,
    solve_weighted_poisson_source_physical_direct_batch,
    weighted_laplacian,
)


def test_weighted_poisson_gradient_through_q_operator_and_rhs():
    cfg = PoissonConfig(dx=0.25, cg_tol=1e-12, cg_maxiter=300)
    q0 = jnp.asarray([
        [1.0, 0.9, 0.8, 0.7],
        [0.95, 0.85, 0.75, 0.65],
        [0.9, 0.8, 0.7, 0.6],
        [0.85, 0.75, 0.65, 0.55],
    ], dtype=jnp.float64)
    h = jnp.asarray([
        [0.3, -0.2, 0.1, -0.1],
        [-0.1, 0.2, -0.2, 0.1],
        [0.15, -0.05, 0.08, -0.12],
        [-0.2, 0.1, -0.05, 0.1],
    ], dtype=jnp.float64)
    # Center h under q so the physical RHS has zero total mass.
    h = h - jnp.sum(q0 * h) / jnp.sum(q0)
    direction = jnp.linspace(-0.15, 0.12, q0.size).reshape(q0.shape)

    def objective(theta):
        q = q0 * jnp.exp(theta * direction)
        centered_h = h - jnp.sum(q * h) / jnp.sum(q)
        # q changes both K(q) and b(q,h).
        return solve_weighted_poisson(q, centered_h, cfg).action

    theta = jnp.asarray(0.0, dtype=jnp.float64)
    grad = jax.grad(objective)(theta)
    eps = 1e-5
    fd = (objective(theta + eps) - objective(theta - eps)) / (2 * eps)
    assert jnp.isfinite(grad)
    assert jnp.allclose(grad, fd, rtol=2e-4, atol=2e-6)


def test_physical_direct_solver_recovers_connected_manufactured_solution():
    ny, nx, dx = 7, 8, 0.2
    yy, xx = np.mgrid[:ny, :nx]
    q = 0.4 + 0.03 * xx + 0.02 * yy
    expected = np.cos(np.pi * (xx + 0.5) / nx) * np.cos(
        2.0 * np.pi * (yy + 0.5) / ny
    )
    expected -= np.sum(q * expected) / np.sum(q)
    source = np.asarray(
        weighted_laplacian(jnp.asarray(expected), jnp.asarray(q), dx)
    ) / q
    cfg = PoissonConfig(dx=dx, operator_floor_rel=1.0e-2, cg_tol=1.0e-12)

    result = solve_weighted_poisson_physical_direct_batch(
        q[None], source[None], cfg, reject_incompatible=True
    )
    source_result = solve_weighted_poisson_source_physical_direct_batch(
        q[None], (q * source)[None], cfg, reject_incompatible=True
    )
    no_floor_result = solve_weighted_poisson_physical_direct_batch(
        q[None],
        source[None],
        PoissonConfig(dx=dx, operator_floor_rel=0.0, cg_tol=1.0e-12),
        reject_incompatible=True,
    )

    assert bool(result.compatible[0])
    assert bool(result.solver_converged[0])
    assert int(result.component_count[0]) == 1
    assert float(result.relative_residual[0]) < 1.0e-11
    np.testing.assert_allclose(result.potential[0], expected, rtol=1.0e-10, atol=1.0e-11)
    np.testing.assert_allclose(
        result.potential, no_floor_result.potential, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(result.potential, source_result.potential, atol=0.0)
    np.testing.assert_allclose(result.action, source_result.action, atol=0.0)
    expected_action = dx**2 * np.sum(
        expected
        * np.asarray(
            weighted_laplacian(jnp.asarray(expected), jnp.asarray(q), dx)
        )
    )
    np.testing.assert_allclose(result.action[0], expected_action, rtol=1.0e-11)


def test_physical_direct_solver_reports_disconnected_incompatible_source():
    q = np.zeros((5, 9), dtype=np.float64)
    q[:, :3] = 1.0
    q[:, 6:] = 1.0
    source = np.zeros_like(q)
    source[:, :3] = 1.0
    source[:, 6:] = -1.0
    cfg = PoissonConfig(dx=0.25, operator_floor_rel=1.0e-2)

    result = solve_weighted_poisson_physical_direct_batch(q[None], source[None], cfg)

    assert not bool(result.compatible[0])
    assert int(result.component_count[0]) == 2
    assert float(result.maximum_component_compatibility_residual[0]) > 0.1
    with pytest.raises(RuntimeError, match="incompatible"):
        solve_weighted_poisson_physical_direct_batch(
            q[None], source[None], cfg, reject_incompatible=True
        )
