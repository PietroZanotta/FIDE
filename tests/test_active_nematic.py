from __future__ import annotations

import json
import numpy as np
import pytest

from experiments.active_nematic.active_nematic_solver import (
    ActiveNematic2D,
    ActiveNematicParams,
)
from experiments.active_nematic.defect_extractor import (
    Defect,
    DefectTracker,
    extract_defects,
    plaquette_winding,
)
from experiments.active_nematic.domain import generate_physical_bank
from experiments.active_nematic.run import _write_merged_manifest


def _periodic_delta(a: float, b: float, period: float) -> float:
    return (a - b + 0.5 * period) % period - 0.5 * period


def _synthetic_periodic_field(
    *, n: int = 128, box_size: float = 2.0 * np.pi, x0: float = 0.73,
    y0: float = 1.11, polarity: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, box_size, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    sx, sy = np.sin(xx - x0), np.sin(yy - y0)
    q1 = 0.5 * (sx * np.cos(polarity) - sy * np.sin(polarity))
    q2 = 0.5 * (sx * np.sin(polarity) + sy * np.cos(polarity))
    return q1, q2


def test_periodic_defect_topology_subgrid_core_and_polarity() -> None:
    box_size = 2.0 * np.pi
    x0, y0, polarity = 0.73, 1.11, 0.7
    q1, q2 = _synthetic_periodic_field(box_size=box_size, x0=x0, y0=y0, polarity=polarity)
    winding = plaquette_winding(q1, q2)
    defects = extract_defects(q1, q2, box_size)
    assert sorted(defect.charge for defect in defects) == [-0.5, -0.5, 0.5, 0.5]
    assert int(winding.sum()) == 0

    defect = min(
        defects,
        key=lambda row: _periodic_delta(row.x, x0, box_size) ** 2
        + _periodic_delta(row.y, y0, box_size) ** 2,
    )
    assert defect.charge == 0.5
    refined_error = np.hypot(
        _periodic_delta(defect.x, x0, box_size),
        _periodic_delta(defect.y, y0, box_size),
    )
    dx = box_size / q1.shape[0]
    center_x = (defect.i + 0.5) * dx
    center_y = (defect.j + 0.5) * dx
    center_error = np.hypot(
        _periodic_delta(center_x, x0, box_size),
        _periodic_delta(center_y, y0, box_size),
    )
    assert refined_error < 0.1 * center_error
    assert defect.polarity is not None
    angular_error = _periodic_delta(defect.polarity, polarity, 2.0 * np.pi)
    assert abs(angular_error) < 2.0e-3
    assert defect.polarity_coherence is not None and defect.polarity_coherence > 0.99


def test_parallel_physical_bank_matches_serial_generation() -> None:
    params = ActiveNematicParams(n=16, dt=0.02)
    seeds = np.asarray([101, 102], dtype=np.int64)
    times = np.asarray([0.0, 0.02, 0.04])
    serial = generate_physical_bank(params, seeds=seeds, times=times, workers=1)
    parallel = generate_physical_bank(params, seeds=seeds, times=times, workers=2)
    np.testing.assert_array_equal(parallel.q1, serial.q1)
    np.testing.assert_array_equal(parallel.q2, serial.q2)


def test_reference_manifest_merge_preserves_other_seeds(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    _write_merged_manifest(
        path,
        [{"reference_seed": 2, "result": "old-two"}],
    )
    _write_merged_manifest(
        path,
        [
            {"reference_seed": 1, "result": "one"},
            {"reference_seed": 2, "result": "new-two"},
        ],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["runs"] == [
        {"reference_seed": 1, "result": "one"},
        {"reference_seed": 2, "result": "new-two"},
    ]


def test_defect_detection_and_tracking_cross_periodic_seam() -> None:
    box_size = 2.0 * np.pi
    q1, q2 = _synthetic_periodic_field(x0=box_size - 0.03, y0=0.04)
    defects = extract_defects(q1, q2, box_size)
    nearest = min(
        (row for row in defects if row.charge > 0.0),
        key=lambda row: _periodic_delta(row.x, box_size - 0.03, box_size) ** 2
        + _periodic_delta(row.y, 0.04, box_size) ** 2,
    )
    assert abs(_periodic_delta(nearest.x, box_size - 0.03, box_size)) < 1.0e-4
    assert abs(_periodic_delta(nearest.y, 0.04, box_size)) < 1.0e-4

    tracker = DefectTracker(box_size=10.0, max_displacement=0.8)
    first = [Defect(9.8, 2.0, 0.5), Defect(5.0, 5.0, -0.5)]
    event0 = tracker.update(first)
    plus_id, minus_id = first[0].track_id, first[1].track_id
    assert len(event0["births"]) == 2 and not event0["deaths"]
    second = [Defect(0.2, 2.1, 0.5), Defect(8.0, 8.0, -0.5)]
    event1 = tracker.update(second)
    assert second[0].track_id == plus_id
    assert minus_id in event1["deaths"]
    assert second[1].track_id in event1["births"]
    event2 = tracker.update([])
    assert sorted(event2["deaths"]) == sorted(row.track_id for row in second)
    third = [Defect(0.3, 2.1, 0.5)]
    event3 = tracker.update(third)
    assert third[0].track_id in event3["births"]
    assert third[0].track_id != plus_id


def test_spectral_derivative_and_screened_stokes_incompressibility() -> None:
    params = ActiveNematicParams(n=16, box_size=2.0 * np.pi, dt=0.01)
    simulation = ActiveNematic2D(params, seed=7)
    derivative = simulation.derivative(np.sin(3.0 * simulation.x), axis=0)
    np.testing.assert_allclose(derivative, 3.0 * np.cos(3.0 * simulation.x), atol=2.0e-12)
    ux, uy = simulation.velocity()
    divergence = simulation.derivative(ux, 0) + simulation.derivative(uy, 1)
    assert np.linalg.norm(divergence) <= 5.0e-11 * max(np.linalg.norm(ux) + np.linalg.norm(uy), 1.0)
    pressure = simulation.pressure()
    force_x, force_y = simulation.active_force()
    residual_x = (
        -simulation.derivative(pressure, 0)
        + params.viscosity * simulation.laplacian(ux)
        - params.friction * ux
        + force_x
    )
    residual_y = (
        -simulation.derivative(pressure, 1)
        + params.viscosity * simulation.laplacian(uy)
        - params.friction * uy
        + force_y
    )
    residual_norm = np.hypot(np.linalg.norm(residual_x), np.linalg.norm(residual_y))
    force_norm = np.hypot(np.linalg.norm(force_x), np.linalg.norm(force_y))
    assert residual_norm <= 2.0e-9 * max(force_norm, 1.0)


def test_one_physical_step_is_finite_and_advances_configured_time() -> None:
    params = ActiveNematicParams(n=16, dt=0.005)
    simulation = ActiveNematic2D(params, seed=13)
    simulation.step()
    assert simulation.t == pytest.approx(params.dt)
    assert np.isfinite(simulation.q1).all()
    assert np.isfinite(simulation.q2).all()


def test_snapshot_exposes_required_derived_fields() -> None:
    simulation = ActiveNematic2D(ActiveNematicParams(n=16), seed=11)
    snapshot = simulation.snapshot()
    required = {
        "t", "q1", "q2", "S", "theta", "u_x", "u_y", "speed", "vorticity",
        "pressure", "active_force_x", "active_force_y", "H1", "H2",
    }
    assert required <= snapshot.keys()
    np.testing.assert_allclose(snapshot["speed"], np.hypot(snapshot["u_x"], snapshot["u_y"]))
