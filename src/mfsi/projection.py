from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class IProjectionConfig:
    """Forward root-solve numerics.

    Reverse mode never differentiates through Newton iterations.  It uses the
    implicit derivative of the converged moment-matching equation.
    """

    max_steps: int = 300
    residual_tol: float = 1.0e-10
    newton_ridge: float = 1.0e-9
    step_cap: float = 20.0
    lambda_clip: float = 1000.0
    line_search_steps: int = 8
    implicit_ridge: float = 0.0


class IProjectionState(NamedTuple):
    lam: Array
    weights: Array
    moments: Array
    residual: Array
    covariance: Array
    ess_fraction: Array


class IProjectionTrajectoryState(NamedTuple):
    """Batched states with leading shape `[trial,time]`."""

    lam: Array
    weights: Array
    moments: Array
    residual: Array
    covariance: Array
    ess_fraction: Array


def normalized_weights(phi: Array, log_base_weights: Array, lam: Array) -> Array:
    logits = log_base_weights + phi @ lam
    return jax.nn.softmax(logits)


def moment_residual(
    lam: Array,
    phi: Array,
    log_base_weights: Array,
    target: Array,
) -> Array:
    w = normalized_weights(phi, log_base_weights, lam)
    return w @ phi - target


def moment_covariance(phi: Array, weights: Array) -> Array:
    mean = weights @ phi
    centered = phi - mean
    return centered.T @ (weights[:, None] * centered)


def _dual_value(
    lam: Array,
    phi: Array,
    log_base_weights: Array,
    target: Array,
) -> Array:
    logits = log_base_weights + phi @ lam
    return jax.scipy.special.logsumexp(logits) - lam @ target


def _raw_newton_solve(
    phi: Array,
    log_base_weights: Array,
    target: Array,
    lam0: Array,
    cfg: IProjectionConfig,
) -> Array:
    """Damped Newton with early stopping and a warm start.

    The old fixed ``fori_loop(max_steps)`` paid for every Newton iteration even
    after convergence.  This version stops as soon as the calibration residual is
    small, and accepts a previous-time multiplier as ``lam0``.
    """

    m = phi.shape[-1]
    eye = jnp.eye(m, dtype=phi.dtype)
    scales = 0.5 ** jnp.arange(cfg.line_search_steps, dtype=phi.dtype)

    def residual_norm(lam: Array) -> Array:
        return jnp.linalg.norm(moment_residual(lam, phi, log_base_weights, target))

    def cond(state):
        step, lam, resid = state
        return (step < cfg.max_steps) & (resid > cfg.residual_tol)

    def body(state):
        step, lam, _ = state
        w = normalized_weights(phi, log_base_weights, lam)
        residual = w @ phi - target
        cov = moment_covariance(phi, w)
        delta = jnp.linalg.solve(cov + cfg.newton_ridge * eye, residual)

        delta_norm = jnp.linalg.norm(delta)
        scale = jnp.minimum(1.0, cfg.step_cap / jnp.maximum(delta_norm, 1.0e-30))
        delta = scale * delta

        candidates = lam[None, :] - scales[:, None] * delta[None, :]
        candidates = jnp.clip(candidates, -cfg.lambda_clip, cfg.lambda_clip)
        values = jax.vmap(
            lambda x: _dual_value(x, phi, log_base_weights, target)
        )(candidates)
        next_lam = candidates[jnp.argmin(values)]
        return step + 1, next_lam, residual_norm(next_lam)

    lam0 = jnp.clip(jnp.asarray(lam0, dtype=phi.dtype), -cfg.lambda_clip, cfg.lambda_clip)
    init = (jnp.asarray(0, dtype=jnp.int32), lam0, residual_norm(lam0))
    _, lam, _ = jax.lax.while_loop(cond, body, init)
    return lam


def make_i_projection_solver(cfg: IProjectionConfig):
    """Return ``solve(phi, log_base_weights, target, lam0)`` with implicit VJP.

    If ``F(lambda,z)=E_w[Phi]-target=0``, then ``F_lambda=C``.  Given a
    cotangent ``lambda_bar``, reverse mode solves ``C.T @ a=lambda_bar`` and
    propagates ``-(dF/dz).T @ a``.  The warm start is algorithmic only and has
    zero cotangent.
    """

    @jax.custom_vjp
    def solve(
        phi: Array,
        log_base_weights: Array,
        target: Array,
        lam0: Array,
    ) -> Array:
        return _raw_newton_solve(phi, log_base_weights, target, lam0, cfg)

    def fwd(phi, log_base_weights, target, lam0):
        lam = _raw_newton_solve(phi, log_base_weights, target, lam0, cfg)
        return lam, (lam, phi, log_base_weights, target)

    def bwd(saved, bar_lam):
        lam, phi, log_base_weights, target = saved
        w = normalized_weights(phi, log_base_weights, lam)
        cov = moment_covariance(phi, w)
        if cfg.implicit_ridge:
            cov = cov + cfg.implicit_ridge * jnp.eye(cov.shape[0], dtype=cov.dtype)
        adjoint = jnp.linalg.solve(cov.T, bar_lam)

        def fixed_lambda_residual(p, logw, c):
            return moment_residual(lam, p, logw, c)

        _, pullback = jax.vjp(
            fixed_lambda_residual,
            phi,
            log_base_weights,
            target,
        )
        bar_phi, bar_logw, bar_target = pullback(-adjoint)
        return bar_phi, bar_logw, bar_target, jnp.zeros_like(lam)

    solve.defvjp(fwd, bwd)
    return solve


class EmpiricalIProjector:
    def __init__(
        self,
        cfg: IProjectionConfig = IProjectionConfig(),
        *,
        trajectory_backend: str = "jax",
    ):
        if trajectory_backend not in {"jax", "tesseract_cpp"}:
            raise ValueError("trajectory_backend must be 'jax' or 'tesseract_cpp'")
        self.cfg = cfg
        self.trajectory_backend = trajectory_backend
        self.solve_lambda = make_i_projection_solver(cfg)

    def project(
        self,
        phi: Array,
        base_weights: Array,
        target: Array,
        *,
        lam0: Array | None = None,
    ) -> IProjectionState:
        phi = jnp.asarray(phi, dtype=jnp.float64)
        base_weights = jnp.asarray(base_weights, dtype=jnp.float64)
        base_sum = jnp.sum(base_weights)
        base_weights = base_weights / jnp.maximum(base_sum, 1.0e-300)
        # Preserve absolute continuity exactly: zero reference mass must remain zero
        # under every exponential tilt.  The previous 1e-300 floor introduced a tiny
        # artificial support that could be exponentially amplified for large lambda.
        log_base = jnp.where(base_weights > 0.0, jnp.log(base_weights), -jnp.inf)
        target = jnp.asarray(target, dtype=jnp.float64)
        if lam0 is None:
            lam0 = jnp.zeros(target.shape[-1], dtype=jnp.float64)
        else:
            lam0 = jnp.asarray(lam0, dtype=jnp.float64)

        lam = self.solve_lambda(phi, log_base, target, lam0)
        weights = normalized_weights(phi, log_base, lam)
        moments = weights @ phi
        residual = moments - target
        covariance = moment_covariance(phi, weights)
        # Relative ESS must be normalized by the ESS of the actual reference
        # quadrature weights, not by N.  This agrees with the authoritative exact
        # evaluator and is essential for nonuniform Gauss-Hermite banks.
        ess_proj = 1.0 / jnp.maximum(jnp.sum(weights * weights), 1.0e-300)
        ess_base = 1.0 / jnp.maximum(jnp.sum(base_weights * base_weights), 1.0e-300)
        ess_fraction = ess_proj / ess_base
        return IProjectionState(
            lam=lam,
            weights=weights,
            moments=moments,
            residual=residual,
            covariance=covariance,
            ess_fraction=ess_fraction,
        )

    def project_trajectory(
        self,
        phi: Array,
        base_weights: Array,
        targets: Array,
    ) -> IProjectionTrajectoryState:
        """Project a complete `[trial,time]` target bank with warm starts.

        `phi` is `[T,N,M]`, base weights are `[T,N]`, and targets are
        `[B,T,M]`. The warm start links numerical work through time but is
        deliberately absent from the mathematical VJP.
        """
        phi = jnp.asarray(phi, dtype=jnp.float64)
        base_weights = jnp.asarray(base_weights, dtype=jnp.float64)
        targets = jnp.asarray(targets, dtype=jnp.float64)
        if phi.ndim != 3 or base_weights.shape != phi.shape[:2]:
            raise ValueError("phi/base_weights must have shapes [T,N,M]/[T,N]")
        if targets.ndim != 3 or targets.shape[1] != phi.shape[0] or targets.shape[2] != phi.shape[2]:
            raise ValueError("targets must have shape [B,T,M]")

        base_weights = base_weights / jnp.maximum(
            jnp.sum(base_weights, axis=-1, keepdims=True), 1.0e-300
        )
        log_base = jnp.where(base_weights > 0.0, jnp.log(base_weights), -jnp.inf)

        if self.trajectory_backend == "tesseract_cpp":
            from .projection_tesseract import solve_i_projection_trajectory_tesseract

            lam = solve_i_projection_trajectory_tesseract(
                phi, log_base, targets, self.cfg
            )
        else:
            batch = int(targets.shape[0])
            moments = int(targets.shape[-1])

            def step(lam0, xs):
                phi_t, base_t, target_t = xs
                states = jax.vmap(
                    lambda target, warm: self.project(
                        phi_t, base_t, target, lam0=warm
                    )
                )(target_t, lam0)
                return states.lam, states.lam

            _, lam_time_major = jax.lax.scan(
                step,
                jnp.zeros((batch, moments), dtype=jnp.float64),
                (phi, base_weights, jnp.swapaxes(targets, 0, 1)),
            )
            lam = jnp.swapaxes(lam_time_major, 0, 1)

        logits = log_base[None, :, :] + jnp.einsum("tnm,btm->btn", phi, lam)
        weights = jax.nn.softmax(logits, axis=-1)
        projected_moments = jnp.einsum("btn,tnm->btm", weights, phi)
        centered = phi[None, :, :, :] - projected_moments[:, :, None, :]
        covariance = jnp.einsum(
            "btn,btni,btnj->btij", weights, centered, centered
        )
        residual = projected_moments - targets
        ess_projected = 1.0 / jnp.maximum(jnp.sum(weights**2, axis=-1), 1.0e-300)
        ess_base = 1.0 / jnp.maximum(jnp.sum(base_weights**2, axis=-1), 1.0e-300)
        ess_fraction = ess_projected / ess_base[None, :]
        return IProjectionTrajectoryState(
            lam=lam,
            weights=weights,
            moments=projected_moments,
            residual=residual,
            covariance=covariance,
            ess_fraction=ess_fraction,
        )

    def project_candidate_trajectories(
        self,
        phi: Array,
        base_weights: Array,
        targets: Array,
    ) -> IProjectionTrajectoryState:
        """Project candidate-specific feature and target trajectories.

        ``phi`` is ``[C,T,N,M]``, common quadrature weights are ``[T,N]``,
        and targets are ``[C,T,M]``.  Candidate trajectories are independent;
        each multiplier is warm-started only from its own preceding time node.
        """
        phi = jnp.asarray(phi, dtype=jnp.float64)
        base_weights = jnp.asarray(base_weights, dtype=jnp.float64)
        targets = jnp.asarray(targets, dtype=jnp.float64)
        if phi.ndim != 4 or base_weights.shape != phi.shape[1:3]:
            raise ValueError(
                "phi/base_weights must have shapes [C,T,N,M]/[T,N]"
            )
        if targets.shape != (phi.shape[0], phi.shape[1], phi.shape[3]):
            raise ValueError("targets must have shape [C,T,M]")

        base_weights = base_weights / jnp.maximum(
            jnp.sum(base_weights, axis=-1, keepdims=True), 1.0e-300
        )
        log_base = jnp.where(base_weights > 0.0, jnp.log(base_weights), -jnp.inf)

        if self.trajectory_backend == "tesseract_cpp":
            from .projection_tesseract import (
                solve_i_projection_candidate_trajectories_tesseract,
            )

            lam = solve_i_projection_candidate_trajectories_tesseract(
                phi, log_base, targets, self.cfg
            )
        else:
            def one_candidate(phi_candidate, target_candidate):
                state = self.project_trajectory(
                    phi_candidate, base_weights, target_candidate[None, ...]
                )
                return state.lam[0]

            lam = jax.vmap(one_candidate)(phi, targets)

        logits = log_base[None, :, :] + jnp.einsum(
            "ctnm,ctm->ctn", phi, lam
        )
        weights = jax.nn.softmax(logits, axis=-1)
        projected_moments = jnp.einsum("ctn,ctnm->ctm", weights, phi)
        centered = phi - projected_moments[:, :, None, :]
        covariance = jnp.einsum(
            "ctn,ctni,ctnj->ctij", weights, centered, centered
        )
        residual = projected_moments - targets
        ess_projected = 1.0 / jnp.maximum(
            jnp.sum(weights**2, axis=-1), 1.0e-300
        )
        ess_base = 1.0 / jnp.maximum(
            jnp.sum(base_weights**2, axis=-1), 1.0e-300
        )
        return IProjectionTrajectoryState(
            lam=lam,
            weights=weights,
            moments=projected_moments,
            residual=residual,
            covariance=covariance,
            ess_fraction=ess_projected / ess_base[None, :],
        )
