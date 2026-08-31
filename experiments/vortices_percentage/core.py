"""Core numerical definitions for the Vortices Full-action V2 development path.

The particle information projection is intentionally still the frozen V1 hard
moment fiber. V2 uses a direct cell-integrated even-reflection Gaussian for
density/source, its matched odd-normal reflected particle flux, a fixed
physical bandwidth, and the paper sign convention ``K psi = -s`` for
``K=-div(q grad)`` and ``delta=-grad(psi)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
V1_DIR = HERE
for _path in (REPO_ROOT, REPO_ROOT / "src", HERE.parent, V1_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
jax.config.update("jax_enable_x64", True)

from audit_action_decomposition import _load_experiment, _strict_common_artifacts
from bounded_reference import latent_to_physical, physical_to_latent
from experiment import ObservationTrialBank, _smooth_bound_moment_curve
from mfsi.grid import RectangularGrid2D
from mfsi.moments import AnchoredCubicSplineReconstructor
from mfsi.poisson import (
    PoissonConfig,
    solve_weighted_poisson_source_physical_direct_batch,
)
from mfsi.raster import (
    rasterize_projected_particles_reflected_rect,
    reflected_flux_divergence_rect,
    reflected_particle_flux_rect,
)
from mfsi.reference import velocity_mlp


V2_VERSION = "vortices-full-action-v2-reflection-neumann-2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    """Hash every scientific setting through a canonical JSON encoding."""
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_reference_scott_bandwidth(nodes: Any, weights: Any) -> tuple[float, np.ndarray]:
    """Return the median two-dimensional weighted Scott bandwidth.

    This rule depends only on the frozen reference rollout.  In particular, it
    has no grid-spacing floor and cannot depend on a sensor, trial, allowance,
    or action value.
    """
    x = np.asarray(nodes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w, axis=1, keepdims=True)
    mean = np.sum(w[..., None] * x, axis=1)
    variance = np.sum(w[..., None] * (x - mean[:, None, :]) ** 2, axis=1)
    isotropic_scale = np.sqrt(np.mean(variance, axis=1))
    effective_n = 1.0 / np.sum(w * w, axis=1)
    by_time = isotropic_scale * effective_n ** (-1.0 / 6.0)
    return float(np.median(by_time)), by_time


def frozen_common_reference_scott_bandwidth(
    reference_rollouts: list[tuple[Any, Any]],
) -> tuple[float, np.ndarray]:
    """Median of independently qualified reference-only Scott bandwidths.

    This helper prepares the prospective three-reference policy without
    training or loading any additional reference model.  Each list item is a
    ``(nodes, weights)`` pair from one already qualified frozen rollout.
    """
    if not reference_rollouts:
        raise ValueError("at least one qualified reference rollout is required")
    per_reference = np.asarray(
        [frozen_reference_scott_bandwidth(nodes, weights)[0]
         for nodes, weights in reference_rollouts],
        dtype=np.float64,
    )
    return float(np.median(per_reference)), per_reference


@dataclass(frozen=True)
class DevelopmentContext:
    exp: Any
    bank: ObservationTrialBank
    times: np.ndarray
    v1_config: dict[str, Any]
    point_dir: Path
    bank_path: Path
    namespace: int


@dataclass(frozen=True)
class ParticleState:
    eta: np.ndarray
    trial: int
    particle_count: int
    nodes: np.ndarray
    velocity: np.ndarray
    base_weights: np.ndarray
    weights: np.ndarray
    forcing: np.ndarray
    lam: np.ndarray
    lambda_dot: np.ndarray
    moments: np.ndarray
    target: np.ndarray
    target_dot: np.ndarray
    moment_defect: np.ndarray
    calibration_residual: np.ndarray
    ess_fraction: np.ndarray
    covariance_min_eigenvalue: np.ndarray


def load_development_context(
    pareto_dir: Path,
    bank_path: Path,
    *,
    namespace: int,
) -> DevelopmentContext:
    point, result = _strict_common_artifacts(Path(pareto_dir))
    exp, _, times = _load_experiment(point, result["config"])
    with np.load(bank_path, allow_pickle=False) as raw:
        bank = ObservationTrialBank(
            jnp.asarray(raw["sample_indices"], dtype=jnp.int32),
            jnp.asarray(raw["detector_z"], dtype=jnp.float64),
        )
    return DevelopmentContext(
        exp=exp,
        bank=bank,
        times=np.asarray(times, dtype=np.float64),
        v1_config=result["config"],
        point_dir=Path(point),
        bank_path=Path(bank_path),
        namespace=int(namespace),
    )


def hard_fiber_particle_state(
    context: DevelopmentContext,
    eta: Any,
    trial: int,
    *,
    particle_count: int | None = None,
) -> ParticleState:
    """Recalibrate the hard empirical projection on the requested nested prefix."""
    exp = context.exp
    eta_jax = exp.family.canonicalize(jnp.asarray(eta, dtype=jnp.float64))
    eta_np = np.asarray(eta_jax, dtype=np.float64)
    phi_truth = exp.family.features(exp.truth_particles, eta_jax)
    rec = exp._measurement_reconstruction(phi_truth, context.bank, int(trial))

    total_n = int(exp.reference_nodes.shape[1])
    n = total_n if particle_count is None else int(particle_count)
    if not 1 <= n <= total_n:
        raise ValueError(f"particle_count must lie in [1,{total_n}]")
    nodes = np.asarray(exp.reference_nodes[:, :n], dtype=np.float64)
    velocity = np.asarray(exp.reference_velocity[:, :n], dtype=np.float64)
    base = np.asarray(exp.reference_weights[:, :n], dtype=np.float64).copy()
    base /= np.sum(base, axis=1, keepdims=True)
    phi = np.asarray(exp.family.features(jnp.asarray(nodes), eta_jax), dtype=np.float64)
    grad = np.asarray(
        exp.family.feature_gradients(jnp.asarray(nodes), eta_jax), dtype=np.float64
    )

    projection = exp.exact_projector.project_trajectory(
        jnp.asarray(phi), jnp.asarray(base), jnp.asarray(rec.c)[None]
    )
    lam = np.asarray(projection.lam[0], dtype=np.float64)
    weights = np.asarray(projection.weights[0], dtype=np.float64)
    moments = np.asarray(projection.moments[0], dtype=np.float64)
    residual = np.asarray(projection.residual[0], dtype=np.float64)
    covariance = np.asarray(projection.covariance[0], dtype=np.float64)
    ess = np.asarray(projection.ess_fraction[0], dtype=np.float64)

    advective = np.einsum("tnmd,tnd->tnm", grad, velocity)
    mean_advective = np.einsum("tn,tnm->tm", weights, advective)
    g = np.einsum("tnm,tm->tn", advective, lam)
    mean_g = np.einsum("tn,tn->t", weights, g)
    centered = phi - moments[:, None, :]
    cov_phi_g = np.einsum(
        "tn,tnm,tn->tm", weights, centered, g - mean_g[:, None]
    )
    target_dot = np.asarray(rec.c_dot, dtype=np.float64)
    rhs = target_dot - mean_advective - cov_phi_g
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    )
    lambda_dot = np.empty_like(rhs)
    for time_index in range(len(context.times)):
        if float(eigenvalues[time_index, 0]) <= 1.0e-6:
            lambda_dot[time_index] = np.linalg.lstsq(
                covariance[time_index], rhs[time_index], rcond=None
            )[0]
        else:
            lambda_dot[time_index] = np.linalg.solve(
                covariance[time_index], rhs[time_index]
            )
    forcing = np.einsum("tnm,tm->tn", centered, lambda_dot)
    forcing += g - mean_g[:, None]
    forcing -= np.einsum("tn,tn->t", weights, forcing)[:, None]
    moment_defect = target_dot - mean_advective

    return ParticleState(
        eta=eta_np,
        trial=int(trial),
        particle_count=n,
        nodes=nodes,
        velocity=velocity,
        base_weights=base,
        weights=weights,
        forcing=forcing,
        lam=lam,
        lambda_dot=lambda_dot,
        moments=moments,
        target=np.asarray(rec.c, dtype=np.float64),
        target_dot=target_dot,
        moment_defect=moment_defect,
        calibration_residual=np.linalg.norm(residual, axis=-1),
        ess_fraction=ess,
        covariance_min_eigenvalue=eigenvalues[:, 0],
    )


def make_grid(nx: int, ny: int) -> RectangularGrid2D:
    grid = RectangularGrid2D(0.0, 2.0, 0.0, 1.0, int(nx), int(ny))
    grid.require_isotropic_spacing()
    return grid


def rasterize_v2(
    nodes: Any,
    weights: Any,
    forcing: Any,
    grid: RectangularGrid2D,
    *,
    bandwidth: float,
    image_pairs: int = 4,
):
    """Apply the authoritative reflected/Neumann V2 scalar raster."""
    return rasterize_projected_particles_reflected_rect(
        jnp.asarray(nodes, dtype=jnp.float64),
        jnp.asarray(weights, dtype=jnp.float64),
        jnp.asarray(forcing, dtype=jnp.float64),
        grid,
        bandwidth=float(bandwidth),
        image_pairs=int(image_pairs),
    )


@lru_cache(maxsize=32)
def _compiled_reflected_raster(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
    bandwidth: float,
    image_pairs: int,
):
    grid = RectangularGrid2D(x_min, x_max, y_min, y_max, nx, ny)

    @jax.jit
    def one(x, w, f):
        return rasterize_projected_particles_reflected_rect(
            x,
            w,
            f,
            grid,
            bandwidth=bandwidth,
            image_pairs=image_pairs,
        )

    return one


def rasterize_trajectory_v2(
    state: ParticleState,
    grid: RectangularGrid2D,
    *,
    bandwidth: float,
    image_pairs: int = 4,
) -> dict[str, np.ndarray]:
    one = _compiled_reflected_raster(
        float(grid.x_min),
        float(grid.x_max),
        float(grid.y_min),
        float(grid.y_max),
        int(grid.nx),
        int(grid.ny),
        float(bandwidth),
        int(image_pairs),
    )
    rows = [
        one(
            jnp.asarray(state.nodes[index], dtype=jnp.float64),
            jnp.asarray(state.weights[index], dtype=jnp.float64),
            jnp.asarray(state.forcing[index], dtype=jnp.float64),
        )
        for index in range(state.nodes.shape[0])
    ]
    return {
        "q": np.stack([np.asarray(row.q, dtype=np.float64) for row in rows]),
        "mass": np.stack([np.asarray(row.mass, dtype=np.float64) for row in rows]),
        "h": np.stack([np.asarray(row.h, dtype=np.float64) for row in rows]),
        "source": np.stack([np.asarray(row.source, dtype=np.float64) for row in rows]),
        "source_before": np.asarray(
            [row.source_mass_before_center for row in rows], dtype=np.float64
        ),
        "source_after": np.asarray(
            [row.source_mass_after_center for row in rows], dtype=np.float64
        ),
    }


def solve_v2(q: Any, source: Any, grid: RectangularGrid2D):
    """Solve the paper convention ``K psi=-s``; scalar action is sign invariant."""
    cfg = PoissonConfig(
        dx=grid.require_isotropic_spacing(),
        operator_floor_rel=0.0,
        cg_tol=1.0e-10,
        cg_maxiter=4000,
        gauge_strength=0.0,
    )
    q_array = np.asarray(q, dtype=np.float64)
    source_array = np.asarray(source, dtype=np.float64)
    if q_array.ndim == 2:
        q_array = q_array[None]
        source_array = source_array[None]
    return solve_weighted_poisson_source_physical_direct_batch(
        q_array,
        -source_array,
        cfg,
        compatibility_tolerance=1.0e-10,
        reject_incompatible=False,
    )


def edge_energy_density(
    q: np.ndarray, potential: np.ndarray, grid: RectangularGrid2D
) -> tuple[np.ndarray, float]:
    """Allocate the exact arithmetic-edge Dirichlet energy to adjacent cells."""
    q = np.asarray(q, dtype=np.float64)
    psi = np.asarray(potential, dtype=np.float64)
    density = np.zeros_like(q)
    horizontal = 0.5 * (q[:, :-1] + q[:, 1:]) * (
        (psi[:, 1:] - psi[:, :-1]) / grid.dx
    ) ** 2
    vertical = 0.5 * (q[:-1, :] + q[1:, :]) * (
        (psi[1:, :] - psi[:-1, :]) / grid.dy
    ) ** 2
    density[:, :-1] += 0.5 * horizontal
    density[:, 1:] += 0.5 * horizontal
    density[:-1, :] += 0.5 * vertical
    density[1:, :] += 0.5 * vertical
    total = float(np.sum(density) * grid.cell_area)
    return density, total


def top_fraction_share(values: np.ndarray, fraction: float = 0.01) -> float:
    flat = np.sort(np.maximum(np.asarray(values, dtype=np.float64).ravel(), 0.0))
    count = max(1, int(math.ceil(float(fraction) * len(flat))))
    return float(np.sum(flat[-count:]) / max(float(np.sum(flat)), 1.0e-300))


def diagonal_condition_estimate(q: np.ndarray, grid: RectangularGrid2D) -> float:
    """Cheap declared condition proxy: positive finite-volume diagonal range."""
    q = np.asarray(q, dtype=np.float64)
    diag = np.zeros_like(q)
    hx = 0.5 * (q[:, :-1] + q[:, 1:]) / grid.dx**2
    hy = 0.5 * (q[:-1, :] + q[1:, :]) / grid.dy**2
    diag[:, :-1] += hx
    diag[:, 1:] += hx
    diag[:-1, :] += hy
    diag[1:, :] += hy
    positive = diag[diag > 0.0]
    return float(np.max(positive) / np.min(positive))


def independent_poisson(
    q: np.ndarray, source: np.ndarray, grid: RectangularGrid2D
) -> dict[str, Any]:
    """Independent unscaled SciPy incidence/KKT solve of ``K psi=-s``."""
    q = np.asarray(q, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    ny, nx = q.shape
    ids = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    a = np.concatenate([ids[:, :-1].ravel(), ids[:-1, :].ravel()])
    b = np.concatenate([ids[:, 1:].ravel(), ids[1:, :].ravel()])
    conductance = np.concatenate([
        (0.5 * (q[:, :-1] + q[:, 1:]) / grid.dx**2).ravel(),
        (0.5 * (q[:-1, :] + q[1:, :]) / grid.dy**2).ravel(),
    ])
    incidence = sparse.coo_matrix(
        (
            np.tile(np.array([1.0, -1.0]), len(a)),
            (np.repeat(np.arange(len(a)), 2), np.column_stack([a, b]).ravel()),
        ),
        shape=(len(a), nx * ny),
    ).tocsr()
    matrix = (incidence.T @ sparse.diags(conductance) @ incidence).tocsc()
    rhs = -source.ravel()
    gauge = np.ones((nx * ny, 1), dtype=np.float64)
    augmented = sparse.bmat(
        [[matrix, sparse.csc_matrix(gauge)],
         [sparse.csc_matrix(gauge.T), sparse.csc_matrix((1, 1))]],
        format="csc",
    )
    solution = spsolve(augmented, np.r_[rhs, 0.0])[:-1]
    residual = matrix @ solution - rhs
    action = float(grid.cell_area * solution @ (matrix @ solution))
    return {
        "potential": solution.reshape((ny, nx)),
        "action": action,
        "relative_residual": float(
            np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0e-14)
        ),
    }


def weighted_gradient_relative_error(
    q: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    grid: RectangularGrid2D,
) -> float:
    q = np.asarray(q, dtype=np.float64)
    ax = np.sqrt(0.5 * (q[:, :-1] + q[:, 1:])) * np.diff(a, axis=1) / grid.dx
    bx = np.sqrt(0.5 * (q[:, :-1] + q[:, 1:])) * np.diff(b, axis=1) / grid.dx
    ay = np.sqrt(0.5 * (q[:-1, :] + q[1:, :])) * np.diff(a, axis=0) / grid.dy
    by = np.sqrt(0.5 * (q[:-1, :] + q[1:, :])) * np.diff(b, axis=0) / grid.dy
    numerator = np.sqrt(np.sum((ax - bx) ** 2) + np.sum((ay - by) ** 2))
    denominator = np.sqrt(np.sum(ax**2) + np.sum(ay**2))
    return float(numerator / max(denominator, 1.0e-300))


def local_reference_nodes(exp: Any, center_index: int, offset: float) -> np.ndarray:
    """Roll a frozen scientific-grid state locally in latent coordinates."""
    x0 = jnp.asarray(exp.reference_nodes[center_index], dtype=jnp.float64)
    z0 = physical_to_latent(x0, eps=exp.reference.transform_eps)
    t0 = float(np.asarray(exp.times)[center_index])
    steps = max(4, int(math.ceil(abs(offset) / (0.05 / 16.0))))
    dt = float(offset) / steps

    def body(i, z):
        t = t0 + i.astype(jnp.float64) * dt
        k1 = velocity_mlp(exp.reference.params, t, z)
        k2 = velocity_mlp(exp.reference.params, t + 0.5 * dt, z + 0.5 * dt * k1)
        k3 = velocity_mlp(exp.reference.params, t + 0.5 * dt, z + 0.5 * dt * k2)
        k4 = velocity_mlp(exp.reference.params, t + dt, z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return np.asarray(latent_to_physical(jax.lax.fori_loop(0, steps, body, z0)))


def reconstructed_moments_at_times(
    context: DevelopmentContext,
    eta: Any,
    trial: int,
    evaluation_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the actual noisy bounded reconstruction on arbitrary times."""
    exp = context.exp
    eta_jax = exp.family.canonicalize(jnp.asarray(eta, dtype=jnp.float64))
    phi_truth = exp.family.features(exp.truth_particles, eta_jax)
    phi_acq = phi_truth[exp.acq_idx]
    indices = context.bank.sample_indices[int(trial)]
    detector = context.bank.detector_z[int(trial)]
    sampled = jax.vmap(lambda p, ii: jnp.mean(p[ii], axis=0))(phi_acq, indices)
    exact = jnp.mean(phi_acq, axis=1)
    observations = sampled + float(exp.cfg["measurement"]["obs_noise_std"]) * detector
    endpoint = (exp.acq_idx == 0) | (exp.acq_idx == len(exp.times) - 1)
    observations = jnp.where(endpoint[:, None], exact, observations)
    reconstructor = AnchoredCubicSplineReconstructor(
        np.asarray(exp.times)[np.asarray(exp.acq_idx)],
        np.asarray(evaluation_times, dtype=np.float64),
        exp.spline_cfg,
    )
    fit = reconstructor.reconstruct(observations, exact[0], exact[-1])
    values, derivatives = fit.c, fit.c_dot
    if exp.moment_feature_bounds is not None:
        values, derivatives = _smooth_bound_moment_curve(
            values,
            derivatives,
            *exp.moment_feature_bounds,
            float(exp.moment_bound_transition_width),
        )
    return np.asarray(values, dtype=np.float64), np.asarray(derivatives, dtype=np.float64)


def continuity_check(
    context: DevelopmentContext,
    eta: Any,
    trial: int,
    *,
    grid: RectangularGrid2D,
    bandwidth: float,
    epsilon: float,
    center_index: int = 10,
    state: ParticleState | None = None,
    image_pairs: int = 4,
) -> dict[str, Any]:
    """Centered-difference check of reflected regularized continuity."""
    exp = context.exp
    eta_jax = exp.family.canonicalize(jnp.asarray(eta, dtype=jnp.float64))
    t0 = float(context.times[center_index])
    targets, _ = reconstructed_moments_at_times(
        context,
        eta_jax,
        int(trial),
        np.asarray([t0 - epsilon, t0, t0 + epsilon], dtype=np.float64),
    )
    if state is None:
        state = hard_fiber_particle_state(context, eta_jax, int(trial))
    nodes_minus = local_reference_nodes(exp, center_index, -float(epsilon))
    nodes_center = np.asarray(exp.reference_nodes[center_index], dtype=np.float64)
    nodes_plus = local_reference_nodes(exp, center_index, float(epsilon))
    base = np.asarray(exp.reference_weights[center_index], dtype=np.float64).copy()
    base /= np.sum(base)
    phi_center = np.asarray(exp.family.features(jnp.asarray(nodes_center), eta_jax))
    warm = np.asarray(state.lam[center_index], dtype=np.float64)
    weights = []
    for nodes, target in zip((nodes_minus, nodes_center, nodes_plus), targets):
        phi = np.asarray(exp.family.features(jnp.asarray(nodes), eta_jax))
        projected = exp._exact_tilt(phi, base, target, warm, newton_steps=300)
        weights.append(np.asarray(projected.weights, dtype=np.float64))
        warm = np.asarray(projected.lam, dtype=np.float64)
    q_values = []
    for nodes, weight in zip((nodes_minus, nodes_center, nodes_plus), weights):
        q_values.append(
            np.asarray(
                rasterize_v2(
                    nodes,
                    weight,
                    np.zeros_like(weight),
                    grid,
                    bandwidth=bandwidth,
                    image_pairs=image_pairs,
                ).q,
                dtype=np.float64,
            )
        )
    dq_dt = (q_values[2] - q_values[0]) / (2.0 * float(epsilon))
    weight_center = weights[1]
    velocity = np.asarray(exp.reference_velocity[center_index], dtype=np.float64)
    flux_x, flux_y = reflected_particle_flux_rect(
        nodes_center,
        weight_center,
        velocity,
        grid,
        bandwidth=bandwidth,
        image_pairs=image_pairs,
    )
    divergence = np.asarray(
        reflected_flux_divergence_rect(flux_x, flux_y, grid), dtype=np.float64
    )
    source = np.asarray(
        rasterize_v2(
            nodes_center,
            state.weights[center_index],
            state.forcing[center_index],
            grid,
            bandwidth=bandwidth,
            image_pairs=image_pairs,
        ).source,
        dtype=np.float64,
    )
    estimate = dq_dt + divergence
    difference = estimate - source
    l1 = float(np.sum(np.abs(difference)) * grid.cell_area)
    l2 = float(np.sqrt(np.sum(difference**2) * grid.cell_area))
    source_l2 = float(np.sqrt(np.sum(source**2) * grid.cell_area))
    correlation = float(np.corrcoef(estimate.ravel(), source.ravel())[0, 1])
    points = np.asarray(grid.points(), dtype=np.float64)
    tests = np.stack([
        np.ones(grid.shape),
        points[..., 0],
        points[..., 1],
        np.exp(-((points[..., 0] - 0.5) ** 2 + (points[..., 1] - 0.5) ** 2) / 0.08),
        np.exp(-((points[..., 0] - 1.5) ** 2 + (points[..., 1] - 0.5) ** 2) / 0.08),
    ])
    weak = np.sum(tests * difference[None], axis=(-2, -1)) * grid.cell_area
    weak_scale = np.maximum(
        np.sum(np.abs(tests * source[None]), axis=(-2, -1)) * grid.cell_area,
        1.0e-14,
    )
    particle_moment_source = np.einsum(
        "n,nm,n->m",
        state.weights[center_index],
        phi_center,
        state.forcing[center_index],
    )
    return {
        "epsilon": float(epsilon),
        "grid_nx": int(grid.nx),
        "grid_ny": int(grid.ny),
        "l1_error": l1,
        "l2_error": l2,
        "relative_l2_error": l2 / max(source_l2, 1.0e-14),
        "correlation": correlation,
        "weak_errors": weak.tolist(),
        "weak_relative_errors": (np.abs(weak) / weak_scale).tolist(),
        "maximum_weak_relative_error": float(np.max(np.abs(weak) / weak_scale)),
        "particle_moment_identity_absolute": float(
            np.max(np.abs(particle_moment_source - state.moment_defect[center_index]))
        ),
        "maximum_absolute_normal_boundary_flux": float(
            max(
                np.max(np.abs(np.asarray(flux_x)[:, [0, -1]])),
                np.max(np.abs(np.asarray(flux_y)[[0, -1], :])),
            )
        ),
        "image_pairs": int(image_pairs),
        "flux_semantics": "j_h=S_reflect(q*u); u_h=j_h/q_h",
    }
