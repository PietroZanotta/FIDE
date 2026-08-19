"""Focused specifications for the two-species unbalanced implementation.

Per the implementation brief, these tests are added but are not run as part of
this change.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.active_nematic_unbalanced.defect_extractor import extract_defects
from experiments.active_nematic_unbalanced.periodic_numerics import (
    PeriodicGrid2D,
    PeriodicPoissonConfig,
    periodic_weighted_laplacian,
    solve_periodic_weighted_poisson,
)
from experiments.active_nematic_unbalanced.unbalanced_correction import (
    UnbalancedCorrectionConfig,
    solve_unbalanced_screened_poisson,
    unbalanced_residual,
)
from experiments.active_nematic_unbalanced.unbalanced_reference import (
    FisherRaoPairMassSchedule,
    sample_periodic_kde_bank,
)
from experiments.active_nematic_unbalanced.unbalanced_risk import (
    aggregate_two_species_risk,
    finite_measure_mmd2,
)
from experiments.active_nematic_unbalanced.unbalanced_state import (
    FiniteDefectMeasure,
    TwoSpeciesDefectBank,
    UnbalancedStateConfig,
    reconstruct_coupled_mass_trajectory,
)
from experiments.active_nematic_unbalanced.unbalanced_tangent import (
    append_global_mass_observable,
    unbalanced_tangent_action,
)
from experiments.active_nematic_unbalanced.unbalanced_experiment import (
    _quasi_newton_candidates,
)


def _periodic_delta(a, b, period):
    return (a - b + 0.5 * period) % period - 0.5 * period


@pytest.mark.parametrize("winding,beta", [(1, 0.73), (-1, 4.2)])
def test_signed_texture_phase_recovers_beta(winding: int, beta: float) -> None:
    n, box = 256, 2.0 * np.pi
    x0, y0 = 2.3, 3.7
    coordinates = np.arange(n) * box / n
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    dx = _periodic_delta(xx, x0, box)
    dy = _periodic_delta(yy, y0, box)
    radius = np.hypot(dx, dy)
    phase = winding * np.arctan2(dy, dx) + beta
    amplitude = np.tanh(radius / (2.0 * box / n))
    defects = extract_defects(
        amplitude * np.cos(phase), amplitude * np.sin(phase), box
    )
    expected_charge = 0.5 * winding
    candidates = [row for row in defects if row.charge == expected_charge]
    nearest = min(
        candidates,
        key=lambda row: _periodic_delta(row.x, x0, box) ** 2
        + _periodic_delta(row.y, y0, box) ** 2,
    )
    assert nearest.orientation_coherence > 0.98
    assert abs(_periodic_delta(nearest.orientation_phase_beta, beta, 2.0 * np.pi)) < 0.03
    if winding > 0:
        assert abs(_periodic_delta(nearest.polarity, beta, 2.0 * np.pi)) < 0.03
    else:
        expected_arm = (beta / 3.0) % (2.0 * np.pi / 3.0)
        assert abs(_periodic_delta(nearest.triatic_arm_angle, expected_arm, 2.0 * np.pi / 3.0)) < 0.01


def _bank() -> TwoSpeciesDefectBank:
    plus_states = np.asarray([[1.0, 1.0, 0.1], [2.0, 2.0, 0.2], [3.0, 3.0, 0.3]])
    minus_states = np.asarray([[1.5, 1.5, 0.4], [2.5, 2.5, 0.5], [3.5, 3.5, 0.6]])
    return TwoSpeciesDefectBank(
        times=np.asarray([21.0, 31.0]),
        plus_states=plus_states,
        minus_states=minus_states,
        plus_offsets=np.asarray([[0, 1, 2], [2, 2, 3]]),
        minus_offsets=np.asarray([[0, 1, 2], [2, 2, 3]]),
        plus_counts=np.asarray([[1, 1], [0, 1]]),
        minus_counts=np.asarray([[1, 1], [0, 1]]),
        plus_coherence=np.ones(3), minus_coherence=np.ones(3),
        plus_core_residual=np.zeros(3), minus_core_residual=np.zeros(3),
        plus_plaquette=np.zeros((3, 2), dtype=int), minus_plaquette=np.zeros((3, 2), dtype=int),
        rejected_low_coherence_plus=np.zeros((2, 2), dtype=int),
        rejected_low_coherence_minus=np.zeros((2, 2), dtype=int),
        rejected_core_plus=np.zeros((2, 2), dtype=int),
        rejected_core_minus=np.zeros((2, 2), dtype=int),
        box_size=8.0,
        state_config=UnbalancedStateConfig(),
    )


def test_defect_bank_solver_revision_roundtrip(tmp_path) -> None:
    path = tmp_path / "defects.npz"
    _bank().save(path)
    loaded = TwoSpeciesDefectBank.load(path)
    assert loaded.solver_revision == _bank().solver_revision


def test_topological_balance_passes_and_missing_minus_fails() -> None:
    bank = _bank()
    assert bank.charge_balance(run_indices=None, tolerance=0.0).passed
    broken = replace(
        bank,
        minus_states=bank.minus_states[:2],
        minus_offsets=np.asarray([[0, 1, 2], [2, 2, 2]]),
        minus_counts=np.asarray([[1, 1], [0, 0]]),
        minus_coherence=np.ones(2),
        minus_core_residual=np.zeros(2),
        minus_plaquette=np.zeros((2, 2), dtype=int),
    )
    with pytest.raises(ValueError, match="charge-balance.*time=31"):
        broken.charge_balance(run_indices=None, tolerance=0.0)


def test_finite_measure_mass_is_mean_count_not_row_count() -> None:
    measure = _bank().measure("plus", 1, np.asarray([0, 1]))
    assert len(measure.states) == 2
    assert measure.mass == 1.0
    np.testing.assert_allclose(measure.weights, [0.5, 0.5])


def test_fisher_rao_mass_schedule_endpoints_positivity_and_rate() -> None:
    schedule = FisherRaoPairMassSchedule(4.0, 9.0, charge_imbalance=0.0)
    tau = jnp.asarray([0.0, 0.25, 1.0])
    mass = schedule.species_mass("plus", tau)
    rate = schedule.species_source_rate("plus", tau)
    np.testing.assert_allclose(mass[jnp.asarray([0, -1])], [4.0, 9.0])
    assert np.all(np.asarray(mass) > 0.0)
    eps = 1.0e-6
    numerical = (
        np.log(float(schedule.species_mass("plus", 0.25 + eps)))
        - np.log(float(schedule.species_mass("plus", 0.25 - eps)))
    ) / (2.0 * eps)
    np.testing.assert_allclose(rate[1], numerical, rtol=2.0e-6)


def test_periodic_kde_reference_bank_is_reproducible_and_wrapped() -> None:
    states = np.asarray([[0.1, 9.9, 6.2], [9.8, 0.2, 0.1]])
    kwargs = dict(
        sample_count=128,
        seed=71,
        periods=np.asarray([10.0, 10.0, 2.0 * np.pi]),
        position_std=1.0,
        beta_std=0.25,
    )
    first = sample_periodic_kde_bank(states, np.asarray([0.4, 0.6]), **kwargs)
    second = sample_periodic_kde_bank(states, np.asarray([0.4, 0.6]), **kwargs)
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 0.0)
    assert np.all(first < np.asarray([10.0, 10.0, 2.0 * np.pi]))


def test_log_pair_mass_reconstruction_preserves_charge_and_positivity() -> None:
    trajectory = reconstruct_coupled_mass_trajectory(
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([2.1, 3.1, 4.1]),
        np.asarray([1.9, 2.9, 3.9]),
        np.linspace(0.0, 1.0, 9),
        minimum_mass=0.1,
        smoothing=0.0,
        internal_knots=1,
    )
    assert np.all(trajectory.mass_plus > 0.0)
    assert np.all(trajectory.mass_minus > 0.0)
    np.testing.assert_allclose(
        trajectory.mass_plus - trajectory.mass_minus, 0.2, atol=1.0e-12
    )
    np.testing.assert_allclose(
        trajectory.relative_rate_plus,
        trajectory.mass_dot_plus / trajectory.mass_plus,
    )


def test_screened_correction_equation_and_continuity_sign() -> None:
    grid = PeriodicGrid2D(2.0 * np.pi, 20)
    coordinate = np.arange(grid.n) * grid.dx
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    q = jnp.asarray((1.0 + 0.15 * np.cos(xx)) / (2.0 * np.pi) ** 2)
    expected = jnp.asarray(0.4 * np.cos(xx) - 0.3 * np.sin(2.0 * yy))
    kappa = 1.7
    h = (
        periodic_weighted_laplacian(expected, q, grid.dx) + q * expected / kappa
    ) / q
    result = solve_unbalanced_screened_poisson(
        q, h, mass=2.3, grid=grid,
        config=UnbalancedCorrectionConfig(reaction_kappa=kappa, cg_tol=1.0e-11, cg_maxiter=500),
    )
    np.testing.assert_allclose(result.potential, expected, rtol=3.0e-8, atol=3.0e-8)
    # div(q delta)-q alpha=-q h, with delta=grad psi and alpha=psi/kappa.
    continuity = -periodic_weighted_laplacian(result.potential, q, grid.dx) - q * result.source_correction + q * h
    assert np.linalg.norm(np.asarray(continuity)) < 1.0e-9
    assert float(result.relative_residual) < 1.0e-9


def test_pure_reaction_has_no_transport() -> None:
    grid = PeriodicGrid2D(2.0 * np.pi, 16)
    q = jnp.full((grid.n, grid.n), 1.0 / (2.0 * np.pi) ** 2)
    kappa, residual, mass = 2.5, 0.7, 3.0
    result = solve_unbalanced_screened_poisson(
        q, jnp.full_like(q, residual), mass=mass, grid=grid,
        config=UnbalancedCorrectionConfig(reaction_kappa=kappa, cg_tol=1.0e-12, cg_maxiter=200),
    )
    np.testing.assert_allclose(result.source_correction, residual, atol=1.0e-10)
    np.testing.assert_allclose(result.move_action, 0.0, atol=1.0e-11)
    np.testing.assert_allclose(result.reaction_action, mass * kappa * residual**2, rtol=2.0e-9)


def test_balanced_limit_matches_legacy_action_without_modifying_legacy_solver() -> None:
    grid = PeriodicGrid2D(2.0 * np.pi, 20)
    coordinate = np.arange(grid.n) * grid.dx
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    q = jnp.full((grid.n, grid.n), 1.0 / (2.0 * np.pi) ** 2)
    h = jnp.asarray(np.cos(xx) + 0.2 * np.sin(yy))
    old = solve_periodic_weighted_poisson(
        q, h, grid, PeriodicPoissonConfig(operator_floor_rel=0.0, cg_tol=1.0e-11, cg_maxiter=400)
    )
    new = solve_unbalanced_screened_poisson(
        q, h, mass=1.0, grid=grid,
        config=UnbalancedCorrectionConfig(reaction_kappa=1.0e10, cg_tol=1.0e-11, cg_maxiter=400),
    )
    np.testing.assert_allclose(new.potential, -old.potential, rtol=2.0e-7, atol=2.0e-7)
    np.testing.assert_allclose(new.move_action, old.action, rtol=2.0e-7)
    np.testing.assert_allclose(new.total_action, new.move_action + new.reaction_action, rtol=1.0e-12)


def test_finite_measure_risk_detects_mass_and_is_duplication_invariant() -> None:
    periods = jnp.asarray([8.0, 8.0, 2.0 * np.pi])
    bandwidths = jnp.asarray([0.5, 1.0])
    x = jnp.asarray([[1.0, 2.0, 0.2], [4.0, 3.0, 1.1]])
    first = finite_measure_mmd2(x, jnp.asarray([0.5, 0.5]), x, jnp.asarray([0.5, 0.5]), periods=periods, bandwidths=bandwidths)
    assert float(first.finite_measure_risk) < 1.0e-12
    changed_mass = finite_measure_mmd2(x, jnp.asarray([1.0, 1.0]), x, jnp.asarray([0.5, 0.5]), periods=periods, bandwidths=bandwidths)
    assert float(changed_mass.finite_measure_risk) > 0.0
    duplicated = finite_measure_mmd2(
        jnp.repeat(x, 2, axis=0), jnp.full(4, 0.25), x, jnp.asarray([0.5, 0.5]),
        periods=periods, bandwidths=bandwidths,
    )
    np.testing.assert_allclose(duplicated.finite_measure_risk, first.finite_measure_risk, atol=1.0e-12)


def test_two_species_risk_uses_declared_weights() -> None:
    periods = jnp.asarray([8.0, 8.0, 2.0 * np.pi])
    x = jnp.asarray([[1.0, 2.0, 0.2]])
    plus = finite_measure_mmd2(x, jnp.asarray([2.0]), x, jnp.asarray([1.0]), periods=periods, bandwidths=jnp.asarray([1.0]))
    minus = finite_measure_mmd2(x, jnp.asarray([3.0]), x, jnp.asarray([1.0]), periods=periods, bandwidths=jnp.asarray([1.0]))
    total = aggregate_two_species_risk(plus, minus, weight_plus=2.0, weight_minus=0.5)
    np.testing.assert_allclose(total.total, 2.0 * plus.finite_measure_risk + 0.5 * minus.finite_measure_risk)


def test_unbalanced_tangent_decomposes_and_global_mass_is_auxiliary() -> None:
    phi = jnp.asarray([[0.2], [0.8]])
    grad = jnp.asarray([[[1.0, 0.0]], [[-1.0, 0.0]]])
    augmented_phi, augmented_grad = append_global_mass_observable(phi, grad)
    result = unbalanced_tangent_action(
        phi=augmented_phi, grad_phi=augmented_grad,
        velocity=jnp.zeros((2, 2)), normalized_weights=jnp.asarray([0.5, 0.5]),
        mass=2.0, target_raw_moments=jnp.asarray([1.0, 2.0]),
        target_raw_moment_dot=jnp.asarray([0.2, 0.3]), reference_source_rate=0.0,
        reaction_kappa=1.5,
    )
    np.testing.assert_allclose(result.total_action, result.transport_action + result.reaction_action, rtol=1.0e-12)
    np.testing.assert_allclose(augmented_grad[:, -1], 0.0)


def test_unbalanced_residual_formula() -> None:
    np.testing.assert_allclose(unbalanced_residual(0.4, 0.3, 0.1), 0.6)


def test_import_isolation_leaves_vortices_config_unchanged() -> None:
    config = Path(__file__).parents[1] / "vortices" / "config.json"
    before = hashlib.sha256(config.read_bytes()).hexdigest()
    __import__("experiments.active_nematic_unbalanced.unbalanced_experiment")
    after = hashlib.sha256(config.read_bytes()).hexdigest()
    assert before == after


def test_exact_law_refinement_retains_seeds_and_improves_smooth_objective() -> None:
    target = jnp.asarray([1.0, 2.0])
    starts = jnp.asarray([[4.0, 5.0], [6.0, 7.0]])
    progress = []
    candidates = _quasi_newton_candidates(
        lambda eta: jnp.sum((eta - target) ** 2),
        starts,
        constraints=((lambda eta: eta[0] - eta[1], 0.0),),
        canonicalize=lambda eta: eta,
        penalty=1.0e4,
        feasibility_tol=1.0e-8,
        maxiter=50,
        gtol=1.0e-10,
        ftol=1.0e-14,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )
    assert len(candidates) == 2 * len(starts)
    np.testing.assert_allclose(candidates[0].eta, starts[0])
    np.testing.assert_allclose(candidates[1].eta, starts[1])
    assert min(row.value for row in candidates if row.feasible) < 1.0e-15
    assert progress == [(1, 2), (2, 2)]


def test_quasi_newton_refinement_projects_back_to_tight_ceiling() -> None:
    candidates = _quasi_newton_candidates(
        lambda eta: (eta[0] - 2.0) ** 2,
        jnp.asarray([[0.0]]),
        constraints=((lambda eta: eta[0], 0.25),),
        canonicalize=lambda eta: eta,
        penalty=1.0,
        feasibility_tol=1.0e-6,
        maxiter=50,
        gtol=1.0e-10,
        ftol=1.0e-14,
    )
    assert candidates[1].feasible
    assert candidates[1].violations[0] <= 1.0e-6
    assert float(candidates[1].eta[0]) > float(candidates[0].eta[0])
    assert float(candidates[1].eta[0]) <= 0.25 + 1.0e-6
