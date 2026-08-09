"""Backend contract shared by local execution and packaged Tesseract APIs."""

from __future__ import annotations

from dataclasses import asdict

from .homometric import PopulationSupport
from .network import PriorParameters
from .solvers import CalibrationResult, TiltedEnsembleResult, calibrate_dual, tilted_ensemble


class LocalSolverBackend:
    kind = "local_jax"

    def run_tilted_ensemble(
        self,
        params: PriorParameters,
        support: PopulationSupport,
        dual: float,
        sampler_options: dict,
        seed: int,
    ) -> TiltedEnsembleResult:
        return tilted_ensemble(
            params, support, dual, seed=seed, **sampler_options
        )

    def run_dual_calibration(
        self,
        params: PriorParameters,
        support: PopulationSupport,
        target_moment: float,
        sampler_options: dict,
        calibration_options: dict,
        seed: int,
    ) -> CalibrationResult:
        return calibrate_dual(
            params,
            support,
            target_moment,
            sampler_options=sampler_options,
            calibration_options=calibration_options,
            seed=seed,
        )


def serialize_tilted(result: TiltedEnsembleResult) -> dict:
    payload = asdict(result)
    for key in ("indices", "weights", "atom_probabilities"):
        payload[key] = payload[key].tolist()
    return payload


def serialize_calibration(result: CalibrationResult) -> dict:
    return {
        "dual": result.dual,
        "status": result.status,
        "converged": result.converged,
        "iterations": result.iterations,
        "sampler_calls": result.sampler_calls,
        "fit_trace": result.fit_trace,
        "residual": result.residual,
        "residual_standard_error": result.residual_standard_error,
        "final_ensemble": serialize_tilted(result.final_ensemble),
    }
