"""Physical-bank and normalized defect-population abstractions.

Raw ``q1,q2`` fields are the physical source of truth.  Positive-defect states
are derived reproducibly and stored raggedly because creation and annihilation
change the number of extant defects.  Sampling from a time slice constructs the
normalized conditional law of a randomly selected extant +1/2 defect; the count
trajectory remains a separate observable and is never interpreted as conserved
probability mass.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

try:  # Support both ``python -m experiments...`` and direct experiment scripts.
    from .active_nematic_solver import (
        ACTIVE_NEMATIC_SOLVER_REVISION,
        ActiveNematic2D,
        ActiveNematicParams,
    )
    from .defect_extractor import extract_defects
except ImportError:  # pragma: no cover - exercised by the direct CLI convention.
    from active_nematic_solver import (
        ACTIVE_NEMATIC_SOLVER_REVISION,
        ActiveNematic2D,
        ActiveNematicParams,
    )
    from defect_extractor import extract_defects

Array = np.ndarray
StateMode = Literal["position", "position_polarity"]


@dataclass(frozen=True)
class PopulationStateConfig:
    """Definition of the normalized positive-defect scientific state."""

    mode: StateMode = "position_polarity"
    min_polarity_coherence: float = 0.0
    fit_rmin_cells: float = 2.0
    fit_rmax_cells: float = 6.0

    def __post_init__(self) -> None:
        if self.mode not in ("position", "position_polarity"):
            raise ValueError("mode must be 'position' or 'position_polarity'")
        if not 0.0 <= self.min_polarity_coherence <= 1.0:
            raise ValueError("min_polarity_coherence must lie in [0,1]")
        if not 0.0 < self.fit_rmin_cells < self.fit_rmax_cells:
            raise ValueError("polarity-fit radii must be positive and ordered")

    @property
    def state_dim(self) -> int:
        return 2 if self.mode == "position" else 3


@dataclass(frozen=True)
class SplitConfig:
    """Disjoint realization split matching the vortices train/design/validation roles."""

    train_runs: int = 32
    design_runs: int = 16
    validation_runs: int = 16
    seed: int = 20260818

    @property
    def total_runs(self) -> int:
        return self.train_runs + self.design_runs + self.validation_runs

    def __post_init__(self) -> None:
        if min(self.train_runs, self.design_runs, self.validation_runs) < 1:
            raise ValueError("every realization split must contain at least one run")


class BankSplit(NamedTuple):
    train: Array
    design: Array
    validation: Array


def make_run_split(config: SplitConfig) -> BankSplit:
    """Return deterministic, disjoint run indices without touching global RNG state."""
    order = np.random.default_rng(config.seed).permutation(config.total_runs)
    a = config.train_runs
    b = a + config.design_runs
    return BankSplit(order[:a], order[a:b], order[b:])


@dataclass(frozen=True)
class PhysicalBank:
    """Raw active-nematic fields with shape ``[run,time,n,n]``."""

    times: Array
    q1: Array
    q2: Array
    seeds: Array
    params: ActiveNematicParams
    solver_revision: str = ACTIVE_NEMATIC_SOLVER_REVISION

    def __post_init__(self) -> None:
        times = np.asarray(self.times)
        q1, q2 = np.asarray(self.q1), np.asarray(self.q2)
        seeds = np.asarray(self.seeds)
        expected = (len(seeds), len(times), self.params.n, self.params.n)
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be a strictly increasing 1-D array")
        if q1.shape != expected or q2.shape != expected:
            raise ValueError(f"q1 and q2 must both have shape {expected}")
        if not (np.isfinite(q1).all() and np.isfinite(q2).all()):
            raise ValueError("physical fields must be finite")
        if self.solver_revision != ACTIVE_NEMATIC_SOLVER_REVISION:
            raise ValueError(
                "physical bank solver revision mismatch: "
                f"bank={self.solver_revision!r}, "
                f"current={ACTIVE_NEMATIC_SOLVER_REVISION!r}"
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray(1, dtype=np.int64),
            times=self.times,
            q1=self.q1,
            q2=self.q2,
            seeds=self.seeds,
            params_json=np.asarray(json.dumps(asdict(self.params), sort_keys=True)),
            solver_revision=np.asarray(self.solver_revision),
        )

    def select_times(self, requested: Array) -> "PhysicalBank":
        """Return an exact configured time subset without rerunning physics."""
        requested = np.asarray(requested, dtype=np.float64)
        if requested.ndim != 1 or len(requested) < 2 or np.any(np.diff(requested) <= 0.0):
            raise ValueError("requested times must be strictly increasing")
        indices = []
        for value in requested:
            matches = np.flatnonzero(
                np.isclose(self.times, value, atol=1.0e-12, rtol=0.0)
            )
            if len(matches) != 1:
                raise ValueError(f"requested population time {value} is absent from physical bank")
            indices.append(int(matches[0]))
        index = np.asarray(indices, dtype=np.int64)
        return PhysicalBank(
            self.times[index],
            self.q1[:, index],
            self.q2[:, index],
            self.seeds,
            self.params,
            self.solver_revision,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PhysicalBank":
        with np.load(Path(path), allow_pickle=False) as data:
            if "solver_revision" not in data:
                raise ValueError(
                    "physical bank has no solver revision and is stale; "
                    "regenerate it with the dealiased ETD2 solver"
                )
            params = ActiveNematicParams(**json.loads(str(data["params_json"].item())))
            return cls(
                data["times"], data["q1"], data["q2"], data["seeds"], params,
                str(data["solver_revision"].item()),
            )


def generate_physical_bank(
    params: ActiveNematicParams,
    *,
    seeds: Array,
    times: Array,
    workers: int = 1,
) -> PhysicalBank:
    """Generate a raw bank from fixed physics and independently seeded initial fields."""
    seeds = np.asarray(seeds, dtype=np.int64)
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or times[0] < 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be increasing, one-dimensional, and start at t >= 0")
    step_numbers = np.rint(times / params.dt).astype(np.int64)
    if not np.allclose(step_numbers * params.dt, times, atol=1.0e-12, rtol=0.0):
        raise ValueError("every saved time must be an integer multiple of dt")
    if int(workers) < 1:
        raise ValueError("workers must be >= 1")

    q1 = np.empty((len(seeds), len(times), params.n, params.n), dtype=np.float64)
    q2 = np.empty_like(q1)
    tasks = [(params, int(seed), step_numbers) for seed in seeds]
    if int(workers) == 1:
        rows = map(_generate_physical_run, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=min(int(workers), len(tasks)))
        rows = executor.map(_generate_physical_run, tasks)
    try:
        for run, (run_q1, run_q2) in enumerate(rows):
            q1[run] = run_q1
            q2[run] = run_q2
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return PhysicalBank(
        times, q1, q2, seeds, params, ACTIVE_NEMATIC_SOLVER_REVISION
    )


def _generate_physical_run(task) -> tuple[Array, Array]:
    """Top-level worker for deterministic parallel realization generation."""
    params, seed, step_numbers = task
    run_q1 = np.empty((len(step_numbers), params.n, params.n), dtype=np.float64)
    run_q2 = np.empty_like(run_q1)
    sim = ActiveNematic2D(params, seed=int(seed))
    previous = 0
    for time_index, step_number in enumerate(step_numbers):
        if step_number > previous:
            sim.step(int(step_number - previous))
        run_q1[time_index] = sim.q1
        run_q2[time_index] = sim.q2
        previous = int(step_number)
    return run_q1, run_q2


@dataclass(frozen=True)
class DefectPopulationBank:
    """Ragged +1/2 defect states and their separate count trajectories.

    ``offsets`` has shape ``[run,time+1]`` and indexes the single ``states``
    array.  Coordinates are periodic on ``[0,box_size)``; polarity, when present,
    is periodic on ``[0,2*pi)``.
    """

    times: Array
    states: Array
    offsets: Array
    counts: Array
    box_size: float
    state_config: PopulationStateConfig

    def __post_init__(self) -> None:
        states = np.asarray(self.states)
        offsets = np.asarray(self.offsets)
        counts = np.asarray(self.counts)
        if states.ndim != 2 or states.shape[1] != self.state_config.state_dim:
            raise ValueError("states have the wrong scientific-state dimension")
        if offsets.ndim != 2 or offsets.shape[1] != len(self.times) + 1:
            raise ValueError("offsets must have shape [run,time+1]")
        if counts.shape != (offsets.shape[0], len(self.times)):
            raise ValueError("counts must have shape [run,time]")
        if not np.array_equal(np.diff(offsets, axis=1), counts):
            raise ValueError("offset differences must equal the count trajectory")

    def samples(self, time_index: int, run_indices: Array | None = None) -> Array:
        """Pool extant positive defects across selected realizations at one time."""
        if run_indices is None:
            run_indices = np.arange(self.offsets.shape[0])
        chunks = [
            self.states[self.offsets[r, time_index] : self.offsets[r, time_index + 1]]
            for r in np.asarray(run_indices, dtype=np.int64)
        ]
        nonempty = [chunk for chunk in chunks if len(chunk)]
        if not nonempty:
            raise ValueError(f"no extant +1/2 defects at time index {time_index}")
        return np.concatenate(nonempty, axis=0)

    def mean_count(self, run_indices: Array | None = None) -> Array:
        rows = self.counts if run_indices is None else self.counts[np.asarray(run_indices)]
        return np.mean(rows, axis=0)

    def resample_trajectory(self, *, run_indices: Array, n: int, seed: int) -> Array:
        """Return ``[time,n,state_dim]`` draws from normalized extant-defect laws.

        Samples at different times are independent population draws and do not
        imply persistent physical defect identity.
        """
        if int(n) < 1:
            raise ValueError("n must be >= 1")
        rng = np.random.default_rng(int(seed))
        rows = []
        for t in range(len(self.times)):
            pool = self.samples(t, run_indices)
            rows.append(pool[rng.integers(0, len(pool), size=int(n))])
        return np.stack(rows)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray(1, dtype=np.int64),
            times=self.times,
            states=self.states,
            offsets=self.offsets,
            counts=self.counts,
            box_size=np.asarray(self.box_size),
            state_config_json=np.asarray(json.dumps(asdict(self.state_config), sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DefectPopulationBank":
        with np.load(Path(path), allow_pickle=False) as data:
            cfg = PopulationStateConfig(**json.loads(str(data["state_config_json"].item())))
            return cls(
                data["times"], data["states"], data["offsets"], data["counts"],
                float(data["box_size"]), cfg,
            )


def extract_population_bank(
    physical: PhysicalBank,
    config: PopulationStateConfig = PopulationStateConfig(),
) -> DefectPopulationBank:
    """Derive the normalized-law source data from raw ``q1,q2`` snapshots."""
    run_count, time_count = physical.q1.shape[:2]
    states: list[list[float]] = []
    counts = np.zeros((run_count, time_count), dtype=np.int64)
    offsets = np.zeros((run_count, time_count + 1), dtype=np.int64)
    cursor = 0
    for run in range(run_count):
        offsets[run, 0] = cursor
        for time_index in range(time_count):
            defects = extract_defects(
                physical.q1[run, time_index],
                physical.q2[run, time_index],
                physical.params.box_size,
                fit_rmin_cells=config.fit_rmin_cells,
                fit_rmax_cells=config.fit_rmax_cells,
            )
            positive = [
                d for d in defects
                if d.charge > 0.0
                and d.polarity is not None
                and d.polarity_coherence is not None
                and d.polarity_coherence >= config.min_polarity_coherence
            ]
            for defect in positive:
                row = [defect.x, defect.y]
                if config.mode == "position_polarity":
                    row.append(float(defect.polarity % (2.0 * np.pi)))
                states.append(row)
            counts[run, time_index] = len(positive)
            cursor += len(positive)
            offsets[run, time_index + 1] = cursor
    state_array = np.asarray(states, dtype=np.float64).reshape((-1, config.state_dim))
    return DefectPopulationBank(
        physical.times, state_array, offsets, counts, physical.params.box_size, config
    )


@dataclass(frozen=True)
class EmpiricalEndpointSource:
    """Shared flow-matching ``EndpointSource`` over normalized defect laws."""

    x0: jax.Array
    x1: jax.Array

    def __post_init__(self) -> None:
        if self.x0.shape != self.x1.shape or self.x0.ndim != 2:
            raise ValueError("x0 and x1 must have matching shape [sample,state_dim]")
        if self.x0.shape[-1] not in (2, 3):
            raise ValueError("active-nematic state dimension must be 2 or 3")

    def sample(self, key: jax.Array, n: int, endpoint: int) -> jax.Array:
        if endpoint not in (0, 1):
            raise ValueError("endpoint must be 0 or 1")
        bank = self.x0 if endpoint == 0 else self.x1
        index = jax.random.randint(key, (int(n),), 0, int(bank.shape[0]))
        return jnp.asarray(bank[index], dtype=jnp.float64)
