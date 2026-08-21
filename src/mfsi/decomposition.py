from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .poisson import weighted_laplacian

Array = jax.Array


class CorrectionDecomposition(NamedTuple):
    """Direct weighted-grid energies for ``delta_hid = delta_* - delta_tan``."""

    full_energy: Array
    tangent_energy: Array
    hidden_energy: Array
    tangent_hidden_inner_product: Array
    discrete_polarization_residual: Array


class RasterTangentProjection(NamedTuple):
    """Minimum-norm moment-feasible correction in the Full raster space."""

    moment_rate_residual: Array
    coefficients: Array
    gram: Array
    gram_rank: Array
    full_moment_residual: Array
    tangent_moment_residual: Array
    hidden_moment_residual: Array
    solver_stabilization_moment_shift: Array
    full_moment_residual_after_stabilization: Array
    full_energy: Array
    tangent_energy: Array
    hidden_energy: Array
    tangent_hidden_inner_product: Array
    pythagorean_residual: Array
    hierarchy_raw_violation: Array


def weighted_dirichlet_inner(
    potential_a: Array,
    potential_b: Array,
    density: Array,
    *,
    dx: float,
    cell_area: float,
) -> Array:
    """Return ``E_q[grad(a) . grad(b)]`` using the Full solver's operator.

    Leading dimensions are preserved.  The last two dimensions are the
    authoritative cell-centered Poisson grid.
    """
    potential_a = jnp.asarray(potential_a, dtype=jnp.float64)
    potential_b = jnp.asarray(potential_b, dtype=jnp.float64)
    density = jnp.asarray(density, dtype=jnp.float64)
    if potential_a.shape != potential_b.shape or potential_a.shape != density.shape:
        raise ValueError("potentials and density must have identical shapes")
    operator_b = jax.vmap(
        lambda b, q: weighted_laplacian(b, q, dx),
        in_axes=(0, 0),
    )(
        potential_b.reshape((-1,) + potential_b.shape[-2:]),
        density.reshape((-1,) + density.shape[-2:]),
    ).reshape(potential_b.shape)
    return float(cell_area) * jnp.sum(
        potential_a * operator_b, axis=(-2, -1)
    )


def correction_decomposition(
    full_potential: Array,
    tangent_potential: Array,
    density: Array,
    *,
    dx: float,
    cell_area: float,
) -> CorrectionDecomposition:
    """Compute Full/Tangent/Hidden energies from the correction potentials.

    Both correction vector fields use the same sign convention, so
    ``delta_hid = delta_* - delta_tan`` is represented by the potential
    difference.  No action is inferred by subtracting two reported scalars.
    """
    hidden_potential = jnp.asarray(full_potential) - jnp.asarray(tangent_potential)
    full_energy = weighted_dirichlet_inner(
        full_potential, full_potential, density, dx=dx, cell_area=cell_area
    )
    tangent_energy = weighted_dirichlet_inner(
        tangent_potential, tangent_potential, density, dx=dx, cell_area=cell_area
    )
    hidden_energy = weighted_dirichlet_inner(
        hidden_potential, hidden_potential, density, dx=dx, cell_area=cell_area
    )
    cross = weighted_dirichlet_inner(
        tangent_potential, hidden_potential, density, dx=dx, cell_area=cell_area
    )
    polarization = full_energy - tangent_energy - hidden_energy - 2.0 * cross
    return CorrectionDecomposition(
        full_energy=full_energy,
        tangent_energy=tangent_energy,
        hidden_energy=hidden_energy,
        tangent_hidden_inner_product=cross,
        discrete_polarization_residual=polarization,
    )


def raster_tangent_projection(
    full_potential: Array,
    density: Array,
    source: Array,
    feature_potentials: Array,
    *,
    dx: float,
    cell_area: float,
    pinv_rcond: float = 1.0e-10,
    operator_floor_rel: float = 0.0,
    gauge_strength: float = 0.0,
    source_is_density: bool = False,
) -> RasterTangentProjection:
    """Project a Full correction onto the raster moment-gradient span.

    The correction represented by a potential ``z`` is ``-grad(z)``.  For
    raster sensor potentials ``phi_j``, the discrete moment-rate map is

    ``L_h(-grad(z))_j = -<grad(phi_j), grad(z)>_{q_h}``.

    The independently rasterized Poisson source defines
    ``r_h,j = sum phi_j s_h dx^2``, where the legacy call convention supplies
    ``h_h`` and sets ``s_h=q_h h_h``.  ``source_is_density=True`` accepts the
    directly deposited ``s_h`` without reconstructing it.  Consequently both the Full and
    Tangent corrections should satisfy ``L_h(delta) = -r_h``.  The Tangent
    potential is ``sum_j c_j phi_j``, with the minimum-norm coefficients
    obtained from ``G_h c = r_h`` using the configured pseudoinverse cutoff.

    Every operator and energy below uses the physical density.  No operator
    floor enters the authoritative Poisson equation; a floor may be used only
    inside a preconditioner.
    Leading dimensions of ``full_potential``, ``density``, and ``source`` are
    preserved; ``feature_potentials`` has shape ``(ny, nx, n_moments)``.
    """
    full_potential = jnp.asarray(full_potential, dtype=jnp.float64)
    density = jnp.asarray(density, dtype=jnp.float64)
    source = jnp.asarray(source, dtype=jnp.float64)
    features = jnp.asarray(feature_potentials, dtype=jnp.float64)
    if full_potential.shape != density.shape or full_potential.shape != source.shape:
        raise ValueError("full potential, density, and source must have identical shapes")
    if features.ndim != 3 or features.shape[:2] != full_potential.shape[-2:]:
        raise ValueError(
            "feature potentials must have shape (ny, nx, n_moments) matching the grid"
        )
    if not 0.0 <= float(pinv_rcond) < 1.0:
        raise ValueError("pinv_rcond must lie in [0, 1)")

    grid_shape = full_potential.shape[-2:]
    leading_shape = full_potential.shape[:-2]
    flat_full = full_potential.reshape((-1,) + grid_shape)
    flat_density = density.reshape((-1,) + grid_shape)
    flat_source = source.reshape((-1,) + grid_shape)
    basis = jnp.moveaxis(features, -1, 0)

    full_operator = jax.vmap(
        lambda z, q: weighted_laplacian(z, q, dx)
    )(flat_full, flat_density)
    basis_operator = jax.vmap(
        lambda q: jax.vmap(lambda phi: weighted_laplacian(phi, q, dx))(basis)
    )(flat_density)

    area = float(cell_area)
    gram = area * jnp.einsum("jxy,nkxy->njk", basis, basis_operator)
    full_pairing = area * jnp.einsum("jxy,nxy->nj", basis, full_operator)
    source_density = flat_source if source_is_density else flat_density * flat_source
    moment_rate_residual = area * jnp.einsum(
        "jxy,nxy->nj", basis, source_density
    )

    # The matrices contain only the small number of sensor moments.  NumPy's
    # SVD gives deterministic rank diagnostics alongside the same Moore-Penrose
    # minimum-norm solution used by the particle evaluator.
    gram_np = jax.device_get(gram)
    residual_np = jax.device_get(moment_rate_residual)
    pinv_np = np.linalg.pinv(gram_np, rcond=float(pinv_rcond))
    coefficients = jnp.asarray(
        np.einsum("nij,nj->ni", pinv_np, residual_np),
        dtype=jnp.float64,
    )
    singular_values = np.linalg.svd(gram_np, compute_uv=False)
    thresholds = float(pinv_rcond) * singular_values[:, :1]
    ranks = jnp.asarray(
        np.sum(singular_values > thresholds, axis=1),
        dtype=jnp.int32,
    )

    tangent_potential = jnp.einsum("nj,jxy->nxy", coefficients, basis)
    decomposition = correction_decomposition(
        flat_full,
        tangent_potential,
        flat_density,
        dx=dx,
        cell_area=cell_area,
    )
    gram_coefficients = jnp.einsum("nij,nj->ni", gram, coefficients)
    full_moment_residual = -full_pairing + moment_rate_residual
    tangent_moment_residual = -gram_coefficients + moment_rate_residual
    hidden_moment_residual = -full_pairing + gram_coefficients

    q_floor = float(operator_floor_rel) * jnp.max(
        flat_density, axis=(-2, -1), keepdims=True
    )
    floor_operator = jax.vmap(
        lambda z, floor: weighted_laplacian(
            z, jnp.ones_like(z) * floor, dx
        )
    )(flat_full, q_floor.reshape((-1,)))
    gauge_vector = flat_density.reshape((flat_density.shape[0], -1))
    gauge_vector = gauge_vector / jnp.maximum(
        jnp.linalg.norm(gauge_vector, axis=1, keepdims=True), 1.0e-300
    )
    gauge_amplitude = float(gauge_strength) * jnp.sum(
        gauge_vector * flat_full.reshape((flat_full.shape[0], -1)), axis=1
    )
    gauge_field = (
        gauge_amplitude[:, None] * gauge_vector
    ).reshape(flat_full.shape)
    stabilization_pairing = area * jnp.einsum(
        "jxy,nxy->nj", basis, floor_operator + gauge_field
    )
    full_residual_after_stabilization = (
        full_moment_residual - stabilization_pairing
    )
    pythagorean = (
        decomposition.full_energy
        - decomposition.tangent_energy
        - decomposition.hidden_energy
    )

    def restore(values: Array, trailing: tuple[int, ...] = ()) -> Array:
        return values.reshape(leading_shape + trailing)

    moment_count = int(features.shape[-1])
    return RasterTangentProjection(
        moment_rate_residual=restore(moment_rate_residual, (moment_count,)),
        coefficients=restore(coefficients, (moment_count,)),
        gram=restore(gram, (moment_count, moment_count)),
        gram_rank=restore(ranks),
        full_moment_residual=restore(full_moment_residual, (moment_count,)),
        tangent_moment_residual=restore(tangent_moment_residual, (moment_count,)),
        hidden_moment_residual=restore(hidden_moment_residual, (moment_count,)),
        solver_stabilization_moment_shift=restore(
            stabilization_pairing, (moment_count,)
        ),
        full_moment_residual_after_stabilization=restore(
            full_residual_after_stabilization, (moment_count,)
        ),
        full_energy=restore(decomposition.full_energy),
        tangent_energy=restore(decomposition.tangent_energy),
        hidden_energy=restore(decomposition.hidden_energy),
        tangent_hidden_inner_product=restore(
            decomposition.tangent_hidden_inner_product
        ),
        pythagorean_residual=restore(pythagorean),
        hierarchy_raw_violation=restore(
            decomposition.tangent_energy - decomposition.full_energy
        ),
    )
