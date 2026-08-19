"""Two-species finite-measure state for the unbalanced active-nematic variant.

Every accepted defect is stored once as ``(x, y, beta)``.  A row has no
intrinsic unit mass: when a measure is formed from ``N_runs`` realizations, each
physical defect receives weight ``1/N_runs``.  Consequently total mass is the
mean defect count per realization and is unaffected by Monte Carlo resampling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np

from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor

try:
    from .defect_extractor import extract_defects
    from .domain import ACTIVE_NEMATIC_SOLVER_REVISION, PhysicalBank
except ImportError:  # pragma: no cover - direct experiment CLI convention.
    from defect_extractor import extract_defects
    from domain import ACTIVE_NEMATIC_SOLVER_REVISION, PhysicalBank


Array = np.ndarray
Species = Literal["plus", "minus"]


@dataclass(frozen=True)
class UnbalancedStateConfig:
    orientation_coherence_min_plus: float = 0.2
    orientation_coherence_min_minus: float = 0.2
    maximum_core_residual: float | None = None
    fit_rmin_cells: float = 2.0
    fit_rmax_cells: float = 6.0

    def __post_init__(self) -> None:
        for value in (
            self.orientation_coherence_min_plus,
            self.orientation_coherence_min_minus,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("orientation coherence thresholds must lie in [0,1]")
        if self.maximum_core_residual is not None and self.maximum_core_residual <= 0.0:
            raise ValueError("maximum_core_residual must be positive when supplied")
        if not 0.0 < self.fit_rmin_cells < self.fit_rmax_cells:
            raise ValueError("texture-fit radii must be positive and ordered")


@dataclass(frozen=True)
class FiniteDefectMeasure:
    states: Array
    weights: Array
    species: Species

    def __post_init__(self) -> None:
        states = np.asarray(self.states)
        weights = np.asarray(self.weights)
        if states.ndim != 2 or states.shape[1] != 3:
            raise ValueError("finite defect states must have shape [sample,3]")
        if weights.shape != (len(states),) or np.any(weights < 0.0):
            raise ValueError("finite-measure weights must be nonnegative and match states")
        if not np.isfinite(states).all() or not np.isfinite(weights).all():
            raise ValueError("finite defect measure must be finite")

    @property
    def mass(self) -> float:
        return float(np.sum(self.weights))

    def normalized_probabilities(self, *, minimum_mass: float) -> Array:
        if self.mass < float(minimum_mass):
            raise ValueError(
                f"{self.species} defect mass {self.mass:g} is below minimum_mass="
                f"{float(minimum_mass):g}"
            )
        return np.asarray(self.weights, dtype=np.float64) / self.mass


@dataclass(frozen=True)
class ChargeBalanceDiagnostics:
    times: Array
    mass_plus: Array
    mass_minus: Array
    imbalance: Array
    expected_imbalance: float
    tolerance: float
    rejected_low_coherence_plus: Array
    rejected_low_coherence_minus: Array
    rejected_core_plus: Array
    rejected_core_minus: Array

    @property
    def maximum_violation(self) -> float:
        return float(
            np.max(np.abs(np.asarray(self.imbalance) - self.expected_imbalance))
        )

    @property
    def passed(self) -> bool:
        return self.maximum_violation <= self.tolerance

    def to_dict(self) -> dict[str, object]:
        return {
            "times": np.asarray(self.times).tolist(),
            "mass_plus": np.asarray(self.mass_plus).tolist(),
            "mass_minus": np.asarray(self.mass_minus).tolist(),
            "imbalance": np.asarray(self.imbalance).tolist(),
            "expected_imbalance": self.expected_imbalance,
            "tolerance": self.tolerance,
            "maximum_violation": self.maximum_violation,
            "passed": self.passed,
            "rejected_low_coherence_plus": np.asarray(
                self.rejected_low_coherence_plus
            ).tolist(),
            "rejected_low_coherence_minus": np.asarray(
                self.rejected_low_coherence_minus
            ).tolist(),
            "rejected_core_plus": np.asarray(self.rejected_core_plus).tolist(),
            "rejected_core_minus": np.asarray(self.rejected_core_minus).tolist(),
        }

    def require(self) -> None:
        if self.passed:
            return
        index = int(
            np.argmax(np.abs(np.asarray(self.imbalance) - self.expected_imbalance))
        )
        raise ValueError(
            "topological charge-balance check failed: "
            f"time={float(self.times[index]):g}, "
            f"M_plus={float(self.mass_plus[index]):g}, "
            f"M_minus={float(self.mass_minus[index]):g}, "
            f"imbalance={float(self.imbalance[index]):g}, "
            f"expected={self.expected_imbalance:g}, tolerance={self.tolerance:g}, "
            "rejected_low_coherence="
            f"(+{int(self.rejected_low_coherence_plus[index])},"
            f"-{int(self.rejected_low_coherence_minus[index])}), "
            "rejected_core="
            f"(+{int(self.rejected_core_plus[index])},"
            f"-{int(self.rejected_core_minus[index])})"
        )


@dataclass(frozen=True)
class TwoSpeciesDefectBank:
    """Ragged accepted states and extraction-quality diagnostics for both signs."""

    times: Array
    plus_states: Array
    minus_states: Array
    plus_offsets: Array
    minus_offsets: Array
    plus_counts: Array
    minus_counts: Array
    plus_coherence: Array
    minus_coherence: Array
    plus_core_residual: Array
    minus_core_residual: Array
    plus_plaquette: Array
    minus_plaquette: Array
    rejected_low_coherence_plus: Array
    rejected_low_coherence_minus: Array
    rejected_core_plus: Array
    rejected_core_minus: Array
    box_size: float
    state_config: UnbalancedStateConfig
    solver_revision: str = ACTIVE_NEMATIC_SOLVER_REVISION

    def __post_init__(self) -> None:
        times = np.asarray(self.times)
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("bank times must be strictly increasing")
        if self.solver_revision != ACTIVE_NEMATIC_SOLVER_REVISION:
            raise ValueError(
                "defect bank solver revision mismatch: "
                f"bank={self.solver_revision!r}, "
                f"current={ACTIVE_NEMATIC_SOLVER_REVISION!r}"
            )
        for species in ("plus", "minus"):
            states = np.asarray(getattr(self, f"{species}_states"))
            offsets = np.asarray(getattr(self, f"{species}_offsets"))
            counts = np.asarray(getattr(self, f"{species}_counts"))
            if states.ndim != 2 or states.shape[1] != 3:
                raise ValueError(f"{species} states must have shape [sample,3]")
            if offsets.ndim != 2 or offsets.shape[1] != len(times) + 1:
                raise ValueError(f"{species} offsets must have shape [run,time+1]")
            if counts.shape != (offsets.shape[0], len(times)):
                raise ValueError(f"{species} counts must have shape [run,time]")
            if not np.array_equal(np.diff(offsets, axis=1), counts):
                raise ValueError(f"{species} offsets do not match counts")
            for suffix in ("coherence", "core_residual"):
                if np.asarray(getattr(self, f"{species}_{suffix}")).shape != (len(states),):
                    raise ValueError(f"{species}_{suffix} must align with states")
            if np.asarray(getattr(self, f"{species}_plaquette")).shape != (len(states), 2):
                raise ValueError(f"{species}_plaquette must have shape [sample,2]")

    @property
    def run_count(self) -> int:
        return int(self.plus_counts.shape[0])

    def _arrays(self, species: Species) -> tuple[Array, Array]:
        if species not in ("plus", "minus"):
            raise ValueError("species must be 'plus' or 'minus'")
        return getattr(self, f"{species}_states"), getattr(self, f"{species}_offsets")

    def measure(
        self,
        species: Species,
        time_index: int,
        run_indices: Array | None = None,
    ) -> FiniteDefectMeasure:
        states, offsets = self._arrays(species)
        runs = (
            np.arange(self.run_count, dtype=np.int64)
            if run_indices is None
            else np.asarray(run_indices, dtype=np.int64)
        )
        if len(runs) == 0:
            raise ValueError("at least one physical realization is required")
        chunks = [states[offsets[r, time_index] : offsets[r, time_index + 1]] for r in runs]
        rows = np.concatenate([row for row in chunks if len(row)], axis=0) if any(len(row) for row in chunks) else np.empty((0, 3))
        weights = np.full(len(rows), 1.0 / len(runs), dtype=np.float64)
        return FiniteDefectMeasure(rows, weights, species)

    def mean_mass(self, species: Species, run_indices: Array | None = None) -> Array:
        counts = getattr(self, f"{species}_counts")
        rows = counts if run_indices is None else counts[np.asarray(run_indices, dtype=np.int64)]
        return np.mean(rows, axis=0)

    def resample_normalized_trajectory(
        self,
        species: Species,
        *,
        run_indices: Array,
        n: int,
        seed: int,
        minimum_mass: float,
    ) -> Array:
        rng = np.random.default_rng(int(seed))
        rows = []
        for time_index in range(len(self.times)):
            measure = self.measure(species, time_index, run_indices)
            probabilities = measure.normalized_probabilities(minimum_mass=minimum_mass)
            rows.append(
                measure.states[
                    rng.choice(len(measure.states), size=int(n), p=probabilities)
                ]
            )
        return np.stack(rows)

    def charge_balance(
        self,
        *,
        run_indices: Array | None,
        tolerance: float,
        expected_imbalance: float = 0.0,
        enforce: bool = True,
    ) -> ChargeBalanceDiagnostics:
        runs = (
            np.arange(self.run_count, dtype=np.int64)
            if run_indices is None
            else np.asarray(run_indices, dtype=np.int64)
        )
        plus = self.mean_mass("plus", runs)
        minus = self.mean_mass("minus", runs)
        diagnostics = ChargeBalanceDiagnostics(
            times=self.times,
            mass_plus=plus,
            mass_minus=minus,
            imbalance=plus - minus,
            expected_imbalance=float(expected_imbalance),
            tolerance=float(tolerance),
            rejected_low_coherence_plus=np.sum(
                self.rejected_low_coherence_plus[runs], axis=0
            ),
            rejected_low_coherence_minus=np.sum(
                self.rejected_low_coherence_minus[runs], axis=0
            ),
            rejected_core_plus=np.sum(self.rejected_core_plus[runs], axis=0),
            rejected_core_minus=np.sum(self.rejected_core_minus[runs], axis=0),
        )
        if enforce:
            diagnostics.require()
        return diagnostics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
            if key not in ("state_config",)
        }
        payload["schema_version"] = np.asarray(2, dtype=np.int64)
        payload["state_config_json"] = np.asarray(
            json.dumps(asdict(self.state_config), sort_keys=True)
        )
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "TwoSpeciesDefectBank":
        with np.load(Path(path), allow_pickle=False) as data:
            if "solver_revision" not in data:
                raise ValueError(
                    "defect bank has no solver revision and is stale; "
                    "regenerate physical-bank and defects"
                )
            config = UnbalancedStateConfig(
                **json.loads(str(data["state_config_json"].item()))
            )
            names = [
                name
                for name in cls.__dataclass_fields__
                if name not in ("state_config",)
            ]
            values = {
                name: (
                    float(data[name])
                    if name == "box_size"
                    else str(data[name].item())
                    if name == "solver_revision"
                    else data[name]
                )
                for name in names
            }
        return cls(**values, state_config=config)


def extract_two_species_bank(
    physical: PhysicalBank,
    config: UnbalancedStateConfig = UnbalancedStateConfig(),
) -> TwoSpeciesDefectBank:
    """Extract both signs once, retaining beta and quality provenance."""
    run_count, time_count = physical.q1.shape[:2]
    states = {"plus": [], "minus": []}
    coherence = {"plus": [], "minus": []}
    residual = {"plus": [], "minus": []}
    plaquette = {"plus": [], "minus": []}
    counts = {
        species: np.zeros((run_count, time_count), dtype=np.int64)
        for species in ("plus", "minus")
    }
    offsets = {
        species: np.zeros((run_count, time_count + 1), dtype=np.int64)
        for species in ("plus", "minus")
    }
    rejected_coherence = {
        species: np.zeros((run_count, time_count), dtype=np.int64)
        for species in ("plus", "minus")
    }
    rejected_core = {
        species: np.zeros((run_count, time_count), dtype=np.int64)
        for species in ("plus", "minus")
    }
    cursor = {"plus": 0, "minus": 0}
    thresholds = {
        "plus": config.orientation_coherence_min_plus,
        "minus": config.orientation_coherence_min_minus,
    }
    for run in range(run_count):
        for species in ("plus", "minus"):
            offsets[species][run, 0] = cursor[species]
        for time_index in range(time_count):
            defects = extract_defects(
                physical.q1[run, time_index],
                physical.q2[run, time_index],
                physical.params.box_size,
                fit_rmin_cells=config.fit_rmin_cells,
                fit_rmax_cells=config.fit_rmax_cells,
            )
            for defect in defects:
                species: Species = "plus" if defect.charge > 0.0 else "minus"
                if (
                    defect.orientation_phase_beta is None
                    or defect.orientation_coherence is None
                    or defect.orientation_coherence < thresholds[species]
                ):
                    rejected_coherence[species][run, time_index] += 1
                    continue
                if (
                    config.maximum_core_residual is not None
                    and defect.core_residual > config.maximum_core_residual
                ):
                    rejected_core[species][run, time_index] += 1
                    continue
                states[species].append(
                    [defect.x, defect.y, defect.orientation_phase_beta]
                )
                coherence[species].append(defect.orientation_coherence)
                residual[species].append(defect.core_residual)
                plaquette[species].append(defect.plaquette_index)
                counts[species][run, time_index] += 1
            for species in ("plus", "minus"):
                cursor[species] += int(counts[species][run, time_index])
                offsets[species][run, time_index + 1] = cursor[species]

    def state_array(species: Species) -> Array:
        return np.asarray(states[species], dtype=np.float64).reshape((-1, 3))

    def scalar_array(values: dict[str, list[float]], species: Species) -> Array:
        return np.asarray(values[species], dtype=np.float64)

    def index_array(species: Species) -> Array:
        return np.asarray(plaquette[species], dtype=np.int64).reshape((-1, 2))

    return TwoSpeciesDefectBank(
        times=physical.times,
        plus_states=state_array("plus"),
        minus_states=state_array("minus"),
        plus_offsets=offsets["plus"],
        minus_offsets=offsets["minus"],
        plus_counts=counts["plus"],
        minus_counts=counts["minus"],
        plus_coherence=scalar_array(coherence, "plus"),
        minus_coherence=scalar_array(coherence, "minus"),
        plus_core_residual=scalar_array(residual, "plus"),
        minus_core_residual=scalar_array(residual, "minus"),
        plus_plaquette=index_array("plus"),
        minus_plaquette=index_array("minus"),
        rejected_low_coherence_plus=rejected_coherence["plus"],
        rejected_low_coherence_minus=rejected_coherence["minus"],
        rejected_core_plus=rejected_core["plus"],
        rejected_core_minus=rejected_core["minus"],
        box_size=physical.params.box_size,
        state_config=config,
        solver_revision=physical.solver_revision,
    )


@dataclass(frozen=True)
class CoupledMassTrajectory:
    mass_plus: Array
    mass_minus: Array
    mass_dot_plus: Array
    mass_dot_minus: Array
    relative_rate_plus: Array
    relative_rate_minus: Array
    pair_mass: Array
    pair_mass_dot: Array
    charge_imbalance: float


def reconstruct_coupled_mass_trajectory(
    observation_times: Array,
    mass_plus_observed: Array,
    mass_minus_observed: Array,
    evaluation_times: Array,
    *,
    minimum_mass: float,
    smoothing: float,
    internal_knots: int = 3,
) -> CoupledMassTrajectory:
    """Smooth log pair mass and reconstruct both species with constant charge.

    All derivatives use the same normalized time variable as the reference flow.
    """
    plus = np.asarray(mass_plus_observed, dtype=np.float64)
    minus = np.asarray(mass_minus_observed, dtype=np.float64)
    if np.any(plus < minimum_mass) or np.any(minus < minimum_mass):
        raise ValueError("observed species mass is below minimum_mass")
    imbalance_rows = plus - minus
    imbalance = float(np.mean(imbalance_rows))
    pair = 0.5 * (plus + minus)
    reconstructor = AnchoredCubicSplineReconstructor(
        observation_times,
        evaluation_times,
        AnchoredCubicSplineConfig(
            internal_knots=int(internal_knots),
            smoothing=float(smoothing),
            ridge_rel=1.0e-10,
            roughness_quadrature_order=8,
        ),
    )
    log_pair = jnp.log(jnp.asarray(pair))[:, None]
    fit = reconstructor.reconstruct(log_pair, log_pair[0], log_pair[-1])
    pair_eval = jnp.exp(fit.c[:, 0])
    pair_dot = pair_eval * fit.c_dot[:, 0]
    mass_plus = pair_eval + 0.5 * imbalance
    mass_minus = pair_eval - 0.5 * imbalance
    if bool(jnp.any(mass_plus < minimum_mass) or jnp.any(mass_minus < minimum_mass)):
        raise ValueError("reconstructed species mass is below minimum_mass")
    return CoupledMassTrajectory(
        mass_plus=np.asarray(mass_plus),
        mass_minus=np.asarray(mass_minus),
        mass_dot_plus=np.asarray(pair_dot),
        mass_dot_minus=np.asarray(pair_dot),
        relative_rate_plus=np.asarray(pair_dot / mass_plus),
        relative_rate_minus=np.asarray(pair_dot / mass_minus),
        pair_mass=np.asarray(pair_eval),
        pair_mass_dot=np.asarray(pair_dot),
        charge_imbalance=imbalance,
    )
