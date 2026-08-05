"""Differentiable ensemble moment projection for periodic particle systems.

The forward solver is a ridge-regularized sequential quadratic programming
(SQP) method for the nearest feasible ensemble.  Each iteration minimizes the
linearized nearest-point objective subject to linearized whitened moment
constraints.  An exact-norm merit function globalizes the step, while fixed
iteration counts and fixed backtracking loops make the baseline directly
differentiable with JAX.  The public API is intentionally compatible with a
later implicit KKT derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .geometry import wrap_positions
from .observables import PairBasis, ensemble_pair_moments


@dataclass(frozen=True)
class ProjectionOptions:
    """Options for the unrolled ensemble moment-projection solver."""

    num_steps: int = 16
    tolerance: float = 1e-8
    kkt_tolerance: float = 1e-6
    ridge: float = 1e-8
    svd_rcond: float = 1e-7
    damping: float = 1.0
    max_step_norm: float = 0.05
    max_correction_norm: float = 0.25
    line_search_steps: int = 10
    line_search_shrink: float = 0.5
    sufficient_decrease: float = 1e-4
    merit_penalty: float = 1.0


def _rms_particle_norm(displacement: Array) -> Array:
    """Root-mean-square Euclidean displacement per particle."""
    return jnp.sqrt(jnp.mean(jnp.sum(displacement * displacement, axis=-1)))


def _clip_global_rms(displacement: Array, maximum: Array) -> tuple[Array, Array]:
    """Clip a displacement by its RMS per-particle norm."""
    norm = _rms_particle_norm(displacement)
    factor = jnp.minimum(1.0, maximum / jnp.maximum(norm, 1e-15))
    return displacement * factor, norm > maximum


def _validate_projection_inputs(
    coordinates: Array,
    target_moments: Array,
    box: Array,
    basis: PairBasis,
    moment_scales: Array,
    basis_mask: Array,
) -> None:
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError(
            f"coordinates must have shape (M, N, 2); got {coordinates.shape}"
        )
    if box.shape != (2,):
        raise ValueError(f"box must have shape (2,), got {box.shape}")
    basis.validate()
    r = basis.centers.shape[0]
    if target_moments.shape != (r,):
        raise ValueError(
            f"target_moments must have shape ({r},); got {target_moments.shape}"
        )
    if moment_scales.shape != (r,):
        raise ValueError(f"moment_scales must have shape ({r},); got {moment_scales.shape}")
    if basis_mask.shape != (r,):
        raise ValueError(f"basis_mask must have shape ({r},); got {basis_mask.shape}")


def _moment_quantities(
    coordinates: Array,
    target_moments: Array,
    box: Array,
    basis: PairBasis,
    moment_scales: Array,
    basis_mask: Array,
) -> tuple[Array, Array, Array]:
    """Return raw moments, whitened active residual, and flattened Jacobian."""

    def raw_moment_fn(value: Array) -> Array:
        return ensemble_pair_moments(value, box, basis)

    moments = raw_moment_fn(coordinates)
    raw_jacobian = jax.jacrev(raw_moment_fn)(coordinates)
    raw_jacobian = raw_jacobian.reshape((moments.shape[0], -1))
    whitening = basis_mask / moment_scales
    residual = whitening * (moments - target_moments)
    jacobian = whitening[:, None] * raw_jacobian
    return moments, residual, jacobian


@partial(
    jax.jit,
    static_argnames=("num_steps", "line_search_steps"),
)
def _project_kernel(
    initial_coordinates: Array,
    target_moments: Array,
    box: Array,
    basis_centers: Array,
    basis_widths: Array,
    moment_scales: Array,
    basis_mask: Array,
    *,
    num_steps: int,
    tolerance: float,
    kkt_tolerance: float,
    ridge: float,
    svd_rcond: float,
    damping: float,
    max_step_norm: float,
    max_correction_norm: float,
    line_search_steps: int,
    line_search_shrink: float,
    sufficient_decrease: float,
    merit_penalty: float,
) -> tuple[Array, dict[str, Array]]:
    """JIT-compatible projection kernel with fixed unrolled control flow."""
    basis = PairBasis(centers=basis_centers, widths=basis_widths)
    initial_wrapped = wrap_positions(initial_coordinates, box)
    # Keep an unwrapped state for a well-defined correction norm and trust region.
    initial_unwrapped = initial_wrapped

    initial_moments, initial_residual, _ = _moment_quantities(
        initial_wrapped,
        target_moments,
        box,
        basis,
        moment_scales,
        basis_mask,
    )
    initial_residual_norm = jnp.linalg.norm(initial_residual)
    initial_correction = jnp.zeros_like(initial_unwrapped)
    initial_objective = 0.5 * jnp.mean(
        jnp.sum(initial_correction * initial_correction, axis=-1)
    )
    initial_merit = initial_objective + merit_penalty * initial_residual_norm
    initial_kkt_norm = jnp.asarray(0.0, dtype=initial_unwrapped.dtype)

    # State: unwrapped coordinates, merit, residual norm, KKT stationarity,
    # iterations, failed searches, correction clips, converged.
    state = (
        initial_unwrapped,
        initial_merit,
        initial_residual_norm,
        initial_kkt_norm,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        (initial_residual_norm <= tolerance) & (initial_kkt_norm <= kkt_tolerance),
    )

    def solver_step(_, state):
        (
            unwrapped,
            merit,
            residual_norm,
            kkt_norm,
            iterations,
            failures,
            clips,
            converged,
        ) = state
        wrapped = wrap_positions(unwrapped, box)
        _, residual, jacobian = _moment_quantities(
            wrapped,
            target_moments,
            box,
            basis,
            moment_scales,
            basis_mask,
        )

        correction = unwrapped - initial_unwrapped
        correction_flat = correction.reshape(-1)
        gram = jacobian @ jacobian.T
        regularized = gram + ridge * jnp.eye(gram.shape[0], dtype=gram.dtype)
        # SQP step for the nearest-point objective under linearized constraints.
        local_dual = jnp.linalg.solve(
            regularized, residual - jacobian @ correction_flat
        )
        raw_step = -damping * (
            correction_flat + jacobian.T @ local_dual
        ).reshape(unwrapped.shape)
        step, step_clipped = _clip_global_rms(raw_step, max_step_norm)

        line_state = (
            jnp.asarray(1.0, dtype=unwrapped.dtype),
            unwrapped,
            merit,
            residual_norm,
            jnp.asarray(False),
            jnp.asarray(False),
        )

        def line_search_step(_, line_state):
            (
                scale,
                best_unwrapped,
                best_merit,
                best_norm,
                accepted,
                total_clipped_any,
            ) = line_state
            candidate_unwrapped = unwrapped + scale * step
            total_correction = candidate_unwrapped - initial_unwrapped
            clipped_total, total_clipped = _clip_global_rms(
                total_correction, max_correction_norm
            )
            candidate_unwrapped = initial_unwrapped + clipped_total
            candidate_wrapped = wrap_positions(candidate_unwrapped, box)
            candidate_moments = ensemble_pair_moments(candidate_wrapped, box, basis)
            candidate_residual = (
                basis_mask * (candidate_moments - target_moments) / moment_scales
            )
            candidate_norm = jnp.linalg.norm(candidate_residual)
            candidate_objective = 0.5 * jnp.mean(
                jnp.sum(clipped_total * clipped_total, axis=-1)
            )
            candidate_merit = candidate_objective + merit_penalty * candidate_norm
            threshold = merit * (1.0 - sufficient_decrease * scale)
            sufficient = candidate_merit <= threshold
            take = (~accepted) & sufficient
            best_unwrapped = jnp.where(take, candidate_unwrapped, best_unwrapped)
            best_merit = jnp.where(take, candidate_merit, best_merit)
            best_norm = jnp.where(take, candidate_norm, best_norm)
            accepted = accepted | sufficient
            total_clipped_any = total_clipped_any | (take & total_clipped)
            scale = jnp.where(accepted, scale, scale * line_search_shrink)
            return (
                scale,
                best_unwrapped,
                best_merit,
                best_norm,
                accepted,
                total_clipped_any,
            )

        (
            _,
            candidate_unwrapped,
            candidate_merit,
            candidate_norm,
            accepted,
            total_clipped,
        ) = jax.lax.fori_loop(0, line_search_steps, line_search_step, line_state)

        # Use the current linearization for the in-loop stationarity test.
        # The exact final KKT diagnostic is recomputed at the returned point.
        candidate_jacobian = jacobian
        candidate_correction = (candidate_unwrapped - initial_unwrapped).reshape(-1)
        candidate_regularized = regularized
        candidate_dual = jnp.linalg.solve(
            candidate_regularized, -(candidate_jacobian @ candidate_correction)
        )
        candidate_stationarity = (
            candidate_correction + candidate_jacobian.T @ candidate_dual
        )
        candidate_kkt_norm = jnp.sqrt(jnp.mean(candidate_stationarity**2))
        newly_converged = (candidate_norm <= tolerance) & (
            candidate_kkt_norm <= kkt_tolerance
        )
        active = ~converged
        unwrapped_out = jnp.where(active, candidate_unwrapped, unwrapped)
        merit_out = jnp.where(active, candidate_merit, merit)
        norm_out = jnp.where(active, candidate_norm, residual_norm)
        kkt_out = jnp.where(active, candidate_kkt_norm, kkt_norm)
        iterations_out = iterations + active.astype(jnp.int32)
        failures_out = failures + (active & (~accepted)).astype(jnp.int32)
        clips_out = clips + (
            active & (step_clipped | total_clipped)
        ).astype(jnp.int32)
        converged_out = converged | (active & newly_converged)
        return (
            unwrapped_out,
            merit_out,
            norm_out,
            kkt_out,
            iterations_out,
            failures_out,
            clips_out,
            converged_out,
        )

    (
        unwrapped,
        final_merit,
        _,
        _,
        iterations,
        failures,
        clips,
        converged,
    ) = jax.lax.fori_loop(0, num_steps, solver_step, state)
    projected = wrap_positions(unwrapped, box)
    final_moments, final_residual, final_jacobian = _moment_quantities(
        projected,
        target_moments,
        box,
        basis,
        moment_scales,
        basis_mask,
    )
    final_gram = final_jacobian @ final_jacobian.T
    gram_eigenvalues = jnp.linalg.eigvalsh(final_gram)
    singular_values = jnp.sqrt(jnp.maximum(gram_eigenvalues, 0.0))[::-1]
    largest = jnp.max(singular_values)
    threshold = svd_rcond * jnp.maximum(largest, jnp.asarray(1e-15, projected.dtype))
    effective_rank = jnp.sum(singular_values > threshold).astype(jnp.int32)
    active_constraints = jnp.sum(basis_mask > 0.0).astype(jnp.int32)
    rank_deficient = effective_rank < active_constraints
    resolved = singular_values > threshold
    smallest = jnp.where(
        jnp.any(resolved),
        jnp.min(jnp.where(resolved, singular_values, jnp.inf)),
        jnp.asarray(0.0, dtype=projected.dtype),
    )
    condition_number = largest / jnp.maximum(
        smallest, jnp.asarray(1e-15, projected.dtype)
    )

    correction = unwrapped - initial_unwrapped
    gram = final_gram
    regularized = gram + ridge * jnp.eye(gram.shape[0], dtype=gram.dtype)
    # KKT multiplier estimate for d + A^T lambda = 0 at the final point.
    dual_whitened = jnp.linalg.solve(
        regularized, -(final_jacobian @ correction.reshape(-1))
    )
    dual_raw = basis_mask * dual_whitened / moment_scales
    stationarity = correction.reshape(-1) + final_jacobian.T @ dual_whitened

    final_residual_norm = jnp.linalg.norm(final_residual)
    diagnostics = {
        "moments_before": initial_moments,
        "moments_after": final_moments,
        "dual_variables": dual_raw,
        "singular_values": singular_values,
        "correction_norm": _rms_particle_norm(correction),
        "constraint_residual_before": initial_residual_norm,
        "constraint_residual": final_residual_norm,
        "projection_objective": 0.5 * jnp.mean(jnp.sum(correction * correction, axis=-1)),
        "merit_before": initial_merit,
        "merit_after": final_merit,
        "kkt_stationarity_norm": jnp.sqrt(jnp.mean(stationarity * stationarity)),
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "condition_number": condition_number,
        "effective_rank": effective_rank,
        "active_constraints": active_constraints,
        "rank_deficient": rank_deficient,
        "iterations": iterations,
        "line_search_failures": failures,
        "correction_clips": clips,
        "converged": converged | (
            (final_residual_norm <= tolerance)
            & (jnp.sqrt(jnp.mean(stationarity * stationarity)) <= kkt_tolerance)
        ),
    }
    return projected, diagnostics


def project_ensemble_moments(
    coordinates: Array,
    target_moments: Array,
    box: Array,
    basis: PairBasis,
    moment_scales: Array | None = None,
    basis_mask: Array | None = None,
    options: ProjectionOptions | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Project an ensemble toward prescribed smooth pair moments.

    Args:
        coordinates: Relaxed ensemble with shape ``(M, N, 2)``.
        target_moments: Raw target pair coefficients with shape ``(R,)``.
        box: Periodic side lengths with shape ``(2,)``.
        basis: Smooth radial pair basis.
        moment_scales: Positive coefficient scales.  Division by these values
            performs diagonal whitening.  Defaults to unit scales.
        basis_mask: Nonnegative ``(R,)`` mask.  A zero explicitly prunes the
            corresponding constraint while preserving fixed output shapes.
        options: Numerical options for the unrolled solver.

    Returns:
        Wrapped projected coordinates and diagnostics.  Only the coordinates
        are intended as a differentiable solver output; diagnostics describe
        the final forward solve and conditioning.
    """
    if options is None:
        options = ProjectionOptions()
    if options.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if options.tolerance < 0 or options.kkt_tolerance < 0:
        raise ValueError("tolerances cannot be negative")
    if options.ridge <= 0:
        raise ValueError("ridge must be positive")
    if options.svd_rcond <= 0:
        raise ValueError("svd_rcond must be positive")
    if options.damping <= 0:
        raise ValueError("damping must be positive")
    if options.max_step_norm <= 0 or options.max_correction_norm <= 0:
        raise ValueError("correction limits must be positive")
    if options.line_search_steps < 1:
        raise ValueError("line_search_steps must be positive")
    if not 0 < options.line_search_shrink < 1:
        raise ValueError("line_search_shrink must be in (0, 1)")
    if not 0 <= options.sufficient_decrease < 1:
        raise ValueError("sufficient_decrease must be in [0, 1)")
    if options.merit_penalty <= 0:
        raise ValueError("merit_penalty must be positive")

    coordinates = jnp.asarray(coordinates)
    dtype = coordinates.dtype
    target_moments = jnp.asarray(target_moments, dtype=dtype)
    box = jnp.asarray(box, dtype=dtype)
    centers = jnp.asarray(basis.centers, dtype=dtype)
    widths = jnp.asarray(basis.widths, dtype=dtype)
    if moment_scales is None:
        moment_scales = jnp.ones_like(target_moments)
    else:
        moment_scales = jnp.asarray(moment_scales, dtype=dtype)
    if basis_mask is None:
        basis_mask = jnp.ones_like(target_moments)
    else:
        basis_mask = jnp.asarray(basis_mask, dtype=dtype)

    normalized_basis = PairBasis(centers=centers, widths=widths)
    _validate_projection_inputs(
        coordinates,
        target_moments,
        box,
        normalized_basis,
        moment_scales,
        basis_mask,
    )
    # Test the predicates, rather than the source arrays, for concreteness. A
    # closed-over device array is not itself a Tracer, but operations on it do
    # produce tracers while this wrapper is staged by an outer jax.jit.
    invalid_scales = jnp.any(moment_scales <= 0)
    if jax.core.is_concrete(invalid_scales) and bool(invalid_scales):
        raise ValueError("moment_scales must be strictly positive")
    invalid_mask = jnp.any(basis_mask < 0)
    if jax.core.is_concrete(invalid_mask) and bool(invalid_mask):
        raise ValueError("basis_mask must be nonnegative")
    no_active_constraints = ~jnp.any(basis_mask > 0)
    if jax.core.is_concrete(no_active_constraints) and bool(no_active_constraints):
        raise ValueError("at least one basis constraint must be active")

    return _project_kernel(
        coordinates,
        target_moments,
        box,
        centers,
        widths,
        moment_scales,
        basis_mask,
        num_steps=options.num_steps,
        tolerance=options.tolerance,
        kkt_tolerance=options.kkt_tolerance,
        ridge=options.ridge,
        svd_rcond=options.svd_rcond,
        damping=options.damping,
        max_step_norm=options.max_step_norm,
        max_correction_norm=options.max_correction_norm,
        line_search_steps=options.line_search_steps,
        line_search_shrink=options.line_search_shrink,
        sufficient_decrease=options.sufficient_decrease,
        merit_penalty=options.merit_penalty,
    )
