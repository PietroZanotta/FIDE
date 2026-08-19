"""Numerical contracts for the isolated dealiased ETD2 physical solver."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from experiments.active_nematic_unbalanced.active_nematic_solver import (
    ACTIVE_NEMATIC_SOLVER_REVISION,
    ActiveNematic2D,
    ActiveNematicParams,
)
from experiments.active_nematic_unbalanced.domain import PhysicalBank


def _state_error(first, second) -> float:
    return float(
        np.sqrt(
            np.mean(
                (first.q1 - second.q1) ** 2
                + (first.q2 - second.q2) ** 2
            )
        )
    )


def test_cubic_dealiasing_matches_independent_four_x_padding() -> None:
    n = 16
    sim = ActiveNematic2D(
        ActiveNematicParams(
            n=n, box_size=8.0, activity=0.0, flow_alignment=0.0
        ),
        seed=2,
    )
    rng = np.random.default_rng(3)
    sim.q1 = sim._project_state(0.1 * rng.standard_normal((n, n)))
    sim.q2 = sim._project_state(0.1 * rng.standard_normal((n, n)))
    actual1, actual2 = sim._nonlinear_rhs_hat(sim.q1, sim.q2)

    padded_n = 4 * n
    scale = (padded_n / n) ** 2
    start = (padded_n - n) // 2

    def pad(spectrum):
        out = np.zeros((padded_n, padded_n), dtype=np.complex128)
        out[start : start + n, start : start + n] = np.fft.fftshift(
            spectrum
        )
        return np.fft.ifftshift(out) * scale

    def truncate(spectrum):
        block = np.fft.fftshift(spectrum)[
            start : start + n, start : start + n
        ]
        return np.fft.ifftshift(block) * sim._state_mask / scale

    q1 = np.fft.ifft2(pad(sim._state_hat(sim.q1))).real
    q2 = np.fft.ifft2(pad(sim._state_hat(sim.q2))).real
    qnorm2 = 2.0 * (q1**2 + q2**2)
    bulk = -(2.0 * sim.p.A + 4.0 * sim.p.C * qnorm2) / sim.p.gamma
    expected1 = truncate(np.fft.fft2(bulk * q1))
    expected2 = truncate(np.fft.fft2(bulk * q2))
    np.testing.assert_allclose(actual1, expected1, rtol=2.0e-13, atol=3.0e-13)
    np.testing.assert_allclose(actual2, expected2, rtol=2.0e-13, atol=3.0e-13)


def test_screened_stokes_residual_and_incompressibility() -> None:
    sim = ActiveNematic2D(
        ActiveNematicParams(n=32, box_size=16.0), seed=4
    )
    u, v = sim.velocity()
    pressure = sim.pressure()
    force_x, force_y = sim.active_force()
    residual_x = (
        sim.p.viscosity * sim.laplacian(u)
        - sim.p.friction * u
        - sim.derivative(pressure, 0)
        - force_x
    )
    residual_y = (
        sim.p.viscosity * sim.laplacian(v)
        - sim.p.friction * v
        - sim.derivative(pressure, 1)
        - force_y
    )
    relative = np.sqrt(np.mean(residual_x**2 + residual_y**2)) / np.sqrt(
        np.mean(force_x**2 + force_y**2)
    )
    divergence = sim.derivative(u, 0) + sim.derivative(v, 1)
    assert relative < 5.0e-13
    assert np.max(np.abs(divergence)) < 5.0e-13


def test_etd2_temporal_self_convergence() -> None:
    base = ActiveNematicParams(n=24, box_size=16.0, activity=0.8)

    def solve(dt):
        sim = ActiveNematic2D(replace(base, dt=dt), seed=17)
        sim.run(0.16)
        return sim

    coarse, medium, fine = solve(0.04), solve(0.02), solve(0.01)
    ratio = _state_error(coarse, medium) / _state_error(medium, fine)
    assert ratio > 3.4


def test_passive_free_energy_decreases() -> None:
    sim = ActiveNematic2D(
        ActiveNematicParams(n=24, box_size=16.0, dt=0.01, activity=0.0),
        seed=19,
    )

    def energy():
        qnorm2 = 2.0 * (sim.q1**2 + sim.q2**2)
        gradients = sum(
            sim.derivative(q, axis) ** 2
            for q in (sim.q1, sim.q2)
            for axis in (0, 1)
        )
        return float(
            np.mean(
                sim.p.A * qnorm2
                + sim.p.C * qnorm2**2
                + 2.0 * sim.p.elastic_L * gradients
            )
        )

    values = [energy()]
    for _ in range(10):
        sim.step()
        values.append(energy())
    assert np.max(np.diff(values)) < 0.0


def test_unversioned_physical_bank_is_rejected(tmp_path) -> None:
    params = ActiveNematicParams(n=8)
    shape = (1, 2, params.n, params.n)
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        times=np.asarray([0.0, params.dt]),
        q1=np.zeros(shape),
        q2=np.zeros(shape),
        seeds=np.asarray([1]),
        params_json=np.asarray(
            __import__("json").dumps(params.__dict__, sort_keys=True)
        ),
    )
    with pytest.raises(ValueError, match="no solver revision.*stale"):
        PhysicalBank.load(path)

    bank = PhysicalBank(
        np.asarray([0.0, params.dt]),
        np.zeros(shape),
        np.zeros(shape),
        np.asarray([1]),
        params,
        ACTIVE_NEMATIC_SOLVER_REVISION,
    )
    current = tmp_path / "current.npz"
    bank.save(current)
    assert PhysicalBank.load(current).solver_revision == ACTIVE_NEMATIC_SOLVER_REVISION
