from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mfsi.projection import EmpiricalIProjector, IProjectionConfig
from mfsi.projection_tesseract import (
    is_tesseract_iprojection_available,
    solve_i_projection_candidate_trajectories_tesseract_forward,
    solve_i_projection_trajectory_tesseract_forward,
    solve_soft_i_projection_trajectory_tesseract_forward,
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


def test_soft_forward_solves_penalized_moment_stationarity():
    phi, base, targets = _inputs()
    cfg = IProjectionConfig(
        max_steps=100,
        residual_tol=1.0e-10,
        newton_ridge=1.0e-12,
        line_search_steps=8,
    )
    log_base = np.log(np.asarray(base))
    moment_count = phi.shape[-1]
    penalty = np.broadcast_to(
        2.5e-3 * np.eye(moment_count),
        (*targets.shape[:2], moment_count, moment_count),
    ).copy()
    result = solve_soft_i_projection_trajectory_tesseract_forward(
        np.asarray(phi), log_base, np.asarray(targets), penalty, cfg
    )
    logits = (
        log_base[None]
        + np.einsum("tnm,btm->btn", np.asarray(phi), result["lambda_values"])
    )
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)
    moments = np.einsum("btn,tnm->btm", weights, np.asarray(phi))
    stationarity = (
        moments
        - np.asarray(targets)
        + np.einsum("btmk,btk->btm", penalty, result["lambda_values"])
    )
    assert np.max(np.linalg.norm(stationarity, axis=-1)) <= cfg.residual_tol
    np.testing.assert_allclose(
        result["hard_moment_residual_norm"],
        np.linalg.norm(moments - np.asarray(targets), axis=-1),
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.all(result["converged"])


def _candidate_inputs():
    phi, base, _ = _inputs()
    candidate_phi = jnp.stack(
        (phi, 0.85 * phi + 0.1 * jnp.sin(phi), 1.1 * phi - 0.05 * jnp.cos(phi))
    )
    key = jax.random.PRNGKey(20260825)
    target_weights = jax.nn.softmax(
        jax.random.normal(key, candidate_phi.shape[:3], dtype=jnp.float64), axis=-1
    )
    targets = jnp.einsum("ctn,ctnm->ctm", target_weights, candidate_phi)
    return candidate_phi, base, targets


def test_candidate_batch_matches_independent_trajectories_and_is_deterministic():
    phi, base, targets = _candidate_inputs()
    cfg = IProjectionConfig(
        max_steps=100,
        residual_tol=1.0e-9,
        newton_ridge=1.0e-9,
        line_search_steps=6,
    )
    log_base = np.log(np.asarray(base))
    actual = solve_i_projection_candidate_trajectories_tesseract_forward(
        np.asarray(phi), log_base, np.asarray(targets), cfg
    )
    repeated = solve_i_projection_candidate_trajectories_tesseract_forward(
        np.asarray(phi), log_base, np.asarray(targets), cfg
    )
    expected = np.stack(
        [
            solve_i_projection_trajectory_tesseract_forward(
                np.asarray(phi[c]), log_base, np.asarray(targets[c : c + 1]), cfg
            )["lambda_values"][0]
            for c in range(phi.shape[0])
        ]
    )
    np.testing.assert_array_equal(actual["lambda_values"], repeated["lambda_values"])
    np.testing.assert_allclose(actual["lambda_values"], expected, rtol=0.0, atol=0.0)
    assert np.all(actual["converged"])
    assert np.max(actual["residual_norm"]) <= cfg.residual_tol


def test_candidate_tesseract_matches_jax_values_vjp_and_jvp():
    phi, base, targets = _candidate_inputs()
    cfg = IProjectionConfig(
        max_steps=100,
        residual_tol=1.0e-9,
        newton_ridge=1.0e-9,
        line_search_steps=6,
    )
    reference = EmpiricalIProjector(cfg, trajectory_backend="jax")
    native = EmpiricalIProjector(cfg, trajectory_backend="tesseract_cpp")
    expected = reference.project_candidate_trajectories(phi, base, targets)
    actual = native.project_candidate_trajectories(phi, base, targets)
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=2e-6, atol=2e-8)

    particle_cotangent = jnp.linspace(-0.2, 0.3, phi.shape[2])

    def loss(projector, p, b, target):
        state = projector.project_candidate_trajectories(p, b, target)
        return (
            jnp.mean(state.weights * particle_cotangent[None, None, :])
            + 0.03 * jnp.mean(state.lam**2)
        )

    expected_grad = jax.grad(
        lambda p, b, t: loss(reference, p, b, t), argnums=(0, 1, 2)
    )(phi, base, targets)
    actual_grad = jax.grad(
        lambda p, b, t: loss(native, p, b, t), argnums=(0, 1, 2)
    )(phi, base, targets)
    for actual_leaf, expected_leaf in zip(actual_grad, expected_grad, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=3e-6, atol=3e-8)

    phi_dot = 0.005 * jnp.cos(phi)
    base_dot = 0.002 * jnp.sin(base)
    target_dot = 0.005 * jnp.sin(targets)
    base_sum = jnp.sum(base, axis=-1, keepdims=True)
    normalized_base = base / base_sum
    normalized_base_dot = (
        base_dot / base_sum
        - base * jnp.sum(base_dot, axis=-1, keepdims=True) / base_sum**2
    )
    log_base_dot = normalized_base_dot / normalized_base
    centered = phi - actual.moments[:, :, None, :]
    mean_phi_dot = jnp.einsum("ctn,ctnm->ctm", actual.weights, phi_dot)
    score_dot = log_base_dot[None, :, :] + jnp.einsum(
        "ctnm,ctm->ctn", phi_dot, actual.lam
    )
    tilt_phi_dot = jnp.einsum(
        "ctn,ctn,ctnm->ctm", actual.weights, score_dot, centered
    )
    expected_jvp = jax.vmap(jax.vmap(jnp.linalg.solve))(
        actual.covariance,
        target_dot - mean_phi_dot - tilt_phi_dot,
    )
    actual_jvp = jax.jvp(
        lambda p, b, t: native.project_candidate_trajectories(p, b, t).lam,
        (phi, base, targets),
        (phi_dot, base_dot, target_dot),
    )[1]
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=3e-6, atol=3e-8)
