#!/usr/bin/env python3
"""Run frozen Stage 3/4 experiment orchestration with a selectable gradient engine.

The scientific drivers remain byte-for-byte unchanged for provenance.  This
runner replaces only their optimizer call with the shared JAX/Tesseract engine.
Tesseract is the default; direct JAX is the reference backend.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from backend_runtime import normalize_backend
from gradient_runtime import run_gradient_engine
import level2_paper_study as paper
import stage3_rollout_adaptation as stage3
import stage3b_confirmatory as stage3b
import stage4_fiber_design as stage4
import stage4b_fiber_design_confirmatory as stage4b


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "stage3": (stage3, "stage3_rollout_adaptation"),
    "stage3b": (stage3b, "stage3b_confirmatory"),
    "stage4": (stage4, "stage4_fiber_design"),
    "stage4b": (stage4b, "stage4b_fiber_design_confirmatory"),
}
CONTROL_CODES = {"full": 0, "stopped_state": 1, "scalar": 2}


def _make_rollout_payload(control, model, gate, raw, target, adaptation, selection):
    payload = {
        **{f"model_{name}": getattr(model, name) for name in paper.MLP._fields},
        "gate": gate,
        "schedule_raw": raw,
        "control": CONTROL_CODES[control],
        "optimizer_steps": stage3.OPTIMIZER_STEPS,
    }
    for prefix, role in (("adaptation", adaptation), ("selection", selection)):
        generation, oracle_bank = role
        oracle_features, oracle_weights = stage3.oracle_projection(
            raw, oracle_bank, target
        )
        payload[f"{prefix}_minus"] = generation[0]
        payload[f"{prefix}_plus"] = generation[1]
        payload[f"{prefix}_noise"] = generation[2]
        payload[f"{prefix}_oracle_features"] = oracle_features
        payload[f"{prefix}_oracle_weights"] = oracle_weights
    return payload


def _rollout_trace(output, steps):
    return [
        {
            "step": index + 1,
            "adaptation_loss": float(output["adaptation_losses"][index]),
            "gradient_norm": float(output["gradient_norms"][index]),
            "parameters": np.asarray(output["parameter_trace"][index]).tolist(),
        }
        for index in range(steps)
    ]


def install_rollout_backend(backend):
    def optimize_modulation(model, gate, raw, target, adaptation, selection):
        started = time.perf_counter()
        output = run_gradient_engine(
            "rollout",
            _make_rollout_payload(
                "full", model, gate, raw, target, adaptation, selection
            ),
            backend,
        )
        selected = jnp.asarray(output["selected_parameters"])
        steps = stage3.OPTIMIZER_STEPS
        trace = _rollout_trace(output, steps)
        for row in trace:
            row["alpha"] = row.pop("parameters")
        return {
            "alpha": selected,
            "trace": trace,
            "candidate_steps": np.asarray(output["candidate_steps"]).tolist(),
            "selection_losses": np.asarray(output["selection_losses"]).tolist(),
            "selected_candidate_index": int(output["selected_candidate_index"]),
            "selected_step": int(output["selected_step"]),
            "initial_adaptation_loss": float(output["initial_adaptation_loss"]),
            "selected_adaptation_loss": float(output["selected_adaptation_loss"]),
            "initial_selection_loss": float(output["initial_selection_loss"]),
            "selected_selection_loss": float(output["selected_selection_loss"]),
            "wall_seconds": time.perf_counter() - started,
        }

    def optimize_control(control, model, gate, raw, target, adaptation, selection):
        started = time.perf_counter()
        output = run_gradient_engine(
            "rollout",
            _make_rollout_payload(
                control, model, gate, raw, target, adaptation, selection
            ),
            backend,
        )
        selected_full = np.asarray(output["selected_parameters"])
        selected = selected_full[:1] if control == "scalar" else selected_full
        trace = _rollout_trace(output, stage3.OPTIMIZER_STEPS)
        if control == "scalar":
            for row in trace:
                row["parameters"] = row["parameters"][:1]
        return {
            "parameters": selected.tolist(),
            "full_alpha": selected_full.tolist(),
            "trace": trace,
            "candidate_steps": np.asarray(output["candidate_steps"]).tolist(),
            "selection_losses": np.asarray(output["selection_losses"]).tolist(),
            "selected_candidate_index": int(output["selected_candidate_index"]),
            "selected_step": int(output["selected_step"]),
            "initial_adaptation_loss": float(output["initial_adaptation_loss"]),
            "selected_adaptation_loss": float(output["selected_adaptation_loss"]),
            "initial_selection_loss": float(output["initial_selection_loss"]),
            "selected_selection_loss": float(output["selected_selection_loss"]),
            "amplitudes": [
                float(stage3.modulation(jnp.asarray(selected_full), jnp.asarray(t), gate))
                for t in (0.0, 0.25, 0.5, 0.75, 1.0)
            ],
            "wall_seconds": time.perf_counter() - started,
        }

    stage3.optimize_modulation = optimize_modulation
    stage3b.optimize_control = optimize_control


def _fiber_payload(raw, geometry, adaptation, selection, stopped, steps):
    return {
        "schedule_raw": raw,
        "common_mean": geometry["common_mean"],
        "theta0": geometry["theta0"],
        "basis": geometry["basis"],
        "adaptation_minus": adaptation[0],
        "adaptation_plus": adaptation[1],
        "adaptation_noise": adaptation[2],
        "selection_minus": selection[0],
        "selection_plus": selection[1],
        "selection_noise": selection[2],
        "stopped": int(stopped),
        "optimizer_steps": steps,
    }


def _fiber_record(output, steps, started):
    return {
        "trace": [
            {
                "step": index + 1,
                "adaptation_objective": float(output["adaptation_objectives"][index]),
                "gradient_norm": float(output["gradient_norms"][index]),
            }
            for index in range(steps)
        ],
        "candidate_steps": np.asarray(output["candidate_steps"]).tolist(),
        "selection_objectives": np.asarray(output["selection_objectives"]).tolist(),
        "selected_candidate_index": int(output["selected_candidate_index"]),
        "selected_step": int(output["selected_step"]),
        "initial_adaptation_objective": float(output["initial_adaptation_objective"]),
        "selected_adaptation_objective": float(output["selected_adaptation_objective"]),
        "initial_selection_objective": float(output["initial_selection_objective"]),
        "selected_selection_objective": float(output["selected_selection_objective"]),
        "wall_seconds": time.perf_counter() - started,
    }


def install_fiber_backend(backend):
    def optimize_observables(raw, geometry, adaptation, selection, steps):
        started = time.perf_counter()
        output = run_gradient_engine(
            "fiber", _fiber_payload(
                raw, geometry, adaptation, selection, False, steps
            ), backend,
        )
        return {
            "theta": jnp.asarray(output["selected_theta"]),
            **_fiber_record(output, steps, started),
        }

    def optimize(raw, geometry, adaptation, selection, stopped):
        started = time.perf_counter()
        output = run_gradient_engine(
            "fiber", _fiber_payload(
                raw, geometry, adaptation, selection, stopped,
                stage4.OPTIMIZER_STEPS,
            ), backend,
        )
        return (
            jnp.asarray(output["selected_theta"]),
            _fiber_record(output, stage4.OPTIMIZER_STEPS, started),
        )

    stage4.optimize_observables = optimize_observables
    stage4b.optimize = optimize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--backend", choices=("jax", "tesseract"), default=None)
    parser.add_argument("--legacy-output", action="store_true")
    args, forwarded = parser.parse_known_args()
    backend = normalize_backend(args.backend)
    module, output_name = EXPERIMENTS[args.experiment]
    install_rollout_backend(backend)
    install_fiber_backend(backend)
    if not args.legacy_output and "--output-dir" not in forwarded:
        output = ROOT / "results" / "backend_runs" / output_name / backend
        forwarded.extend(["--output-dir", str(output)])
        if args.experiment == "stage4b":
            output.mkdir(parents=True, exist_ok=True)
            protocol = output / stage4b.PROTOCOL_NAME
            if not protocol.exists():
                shutil.copy2(ROOT / "stage4b_protocol.json", protocol)
    sys.argv = [f"{args.experiment}.py", *forwarded]
    module.main()


if __name__ == "__main__":
    main()
