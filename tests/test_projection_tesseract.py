from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mfsi.projection import EmpiricalIProjector, IProjectionConfig
from mfsi.projection_tesseract import (
    is_tesseract_iprojection_available,
    solve_i_projection_trajectory_tesseract_forward,
)

jax.config.update("jax_enable_x64", True)


pytestmark = pytest.mark.skipif(
    not is_tesseract_iprojection_available(),
    reason="optional native I-projection Tesseract is not built",
)


def _inputs():
    key = jax.random.PRNGKey(20260816)
    phi = jax.random.normal(key, (4, 96, 3), dtype=jnp.float64)
    base = jnp.broadcast_to(jnp.linspace(0.5, 1.5, 96), (4, 96))
    base = base / jnp.sum(base, axis=-1, keepdims=True)
    target_weights = jax.nn.softmax(
        jax.random.normal(jax.random.fold_in(key, 1), (3, 4, 96), dtype=jnp.float64),
        axis=-1,
    )
    targets = jnp.einsum("btn,tnm->btm", target_weights, phi)
    return phi, base, targets


def test_native_trajectory_matches_jax_forward_and_vjp():
    phi, base, targets = _inputs()
    cfg = IProjectionConfig(
        max_steps=100,
        residual_tol=1.0e-9,
        newton_ridge=1.0e-9,
        line_search_steps=6,
    )
    jax_projector = EmpiricalIProjector(cfg, trajectory_backend="jax")
    native_projector = EmpiricalIProjector(cfg, trajectory_backend="tesseract_cpp")
    expected = jax_projector.project_trajectory(phi, base, targets)
    actual = native_projector.project_trajectory(phi, base, targets)
    np.testing.assert_allclose(actual.lam, expected.lam, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(actual.weights, expected.weights, rtol=2e-7, atol=2e-9)

    particle_cotangent = jnp.linspace(-0.3, 0.4, phi.shape[1])

    def loss(projector, p, target):
        state = projector.project_trajectory(p, base, target)
        return jnp.mean(state.weights * particle_cotangent[None, None, :]) + 0.03 * jnp.mean(state.lam**2)

    expected_grad = jax.grad(lambda p, t: loss(jax_projector, p, t), argnums=(0, 1))(phi, targets)
    actual_grad = jax.grad(lambda p, t: loss(native_projector, p, t), argnums=(0, 1))(phi, targets)
    for actual_leaf, expected_leaf in zip(actual_grad, expected_grad, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=2e-6, atol=2e-8)

    phi_dot = 0.01 * jnp.cos(phi)
    target_dot = 0.01 * jnp.sin(targets)
    # Differentiate the moment equation E_w[phi] = target analytically.  The
    # reference projector is a custom VJP, so JAX deliberately disallows
    # applying forward-mode AD to it.
    centered = phi[None, :, :, :] - actual.moments[:, :, None, :]
    mean_phi_dot = jnp.einsum("btn,tnm->btm", actual.weights, phi_dot)
    score_dot = jnp.einsum("tnm,btm->btn", phi_dot, actual.lam)
    tilt_phi_dot = jnp.einsum(
        "btn,btn,btnm->btm", actual.weights, score_dot, centered
    )
    expected_jvp = jax.vmap(jax.vmap(jnp.linalg.solve))(
        actual.covariance,
        target_dot - mean_phi_dot - tilt_phi_dot,
    )
    actual_jvp = jax.jvp(
        lambda p, t: native_projector.project_trajectory(p, base, t).lam,
        (phi, targets),
        (phi_dot, target_dot),
    )[1]
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2e-6, atol=2e-8)


def test_direct_forward_audit_path_matches_differentiable_native_path():
    phi, base, targets = _inputs()
    cfg = IProjectionConfig(
        max_steps=100,
        residual_tol=1.0e-9,
        newton_ridge=1.0e-9,
        line_search_steps=6,
    )
    expected = EmpiricalIProjector(
        cfg, trajectory_backend="tesseract_cpp"
    ).project_trajectory(phi, base, targets)
    log_base = jnp.where(base > 0.0, jnp.log(base), -jnp.inf)
    actual = solve_i_projection_trajectory_tesseract_forward(
        np.asarray(phi), np.asarray(log_base), np.asarray(targets), cfg
    )
    np.testing.assert_allclose(
        actual["lambda_values"], expected.lam, rtol=2e-7, atol=2e-7
    )
    assert np.all(actual["converged"])
    assert np.max(actual["residual_norm"]) <= cfg.residual_tol
