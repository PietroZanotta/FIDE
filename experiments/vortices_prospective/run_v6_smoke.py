from __future__ import annotations

"""Small end-to-end CUDA qualification of every v6 freeze boundary."""

import copy
from pathlib import Path

import jax

from common import SCRIPT_DIR, load_config, write_json_atomic
from run_v6 import run
from v6_reference_ensemble import v6_paths


def smoke_config():
    cfg = copy.deepcopy(load_config(SCRIPT_DIR / "configs" / "production_v6_common.json"))
    cfg["name"] = "prospective_v6_beta_ablation_smoke"
    cfg["mode"] = "smoke"
    cfg["truth"].update({
        "endpoint_particles": 512,
        "prospective_particles": 512,
        "hidden_validation_particles": 512,
        "endpoint_rk4_substeps": 16,
        "rk4_substeps_per_interval": 4,
    })
    cfg["time"] = {"scientific_nodes": 5, "acquisition_nodes": 5}
    cfg["measurement"]["finite_n"] = 64
    cfg["aggregate_predictor"].update({"grid_nx": 17, "grid_ny": 9, "particle_chunk": 256})
    cfg["moment_reconstruction"]["internal_knots"] = 1
    cfg["reference_training"].update({
        "hidden_width": 16, "hidden_layers": 1, "train_steps": 4,
        "batch_size": 64, "log_every": 2,
    })
    cfg["reference"].update({"particles": 256, "rk4_substeps_per_interval": 2})
    cfg["projection"].update({"max_steps": 100, "backend": "tesseract_cpp"})
    cfg["poisson"].update({"grid_nx": 24, "grid_ny": 12, "cg_tol": 1e-6, "cg_maxiter": 180})
    cfg["validity"].update({
        "max_projection_residual": 2e-5, "min_ess_fraction": 0.005,
        "max_poisson_relative_residual": 2e-6,
    })
    cfg["v4"].update({"selection_crn_trials": 3, "authoritative_crn_trials": 3, "validation_trials": 3})
    cfg["v4"]["gradient_checks"].update({
        "geometries": 1, "multi_trial_count": 1,
        "finite_difference_steps": [0.001, 0.0003], "relative_tolerance": 0.3,
    })
    cfg["v4"]["law_optimizer"].update({"starts": 2, "start_oversample": 32, "crn_trials": 2, "steps": 2, "batch_size": 1})
    cfg["v4"]["full_optimizer"].update({
        "starts": 3, "law_perturbation_starts": 1, "start_oversample": 32,
        "steps": 2, "batch_size": 1,
    })
    cfg["v4"]["tangent_optimizer"].update({
        "starts": 3, "law_perturbation_starts": 1, "start_oversample": 32,
        "crn_trials": 2, "steps": 2, "batch_size": 1,
    })
    cfg["v4"]["full_lbfgs"].update({"max_iterations": 1, "max_line_search_steps": 3})
    cfg["v4"]["funnel"].update({
        "rescore_candidates": 3, "polish_candidates": 1,
        "polish_adam_steps": 1, "polish_batch_size": 1,
        "authoritative_full_finalists": 2,
    })
    for block in cfg["v4"]["full_fidelities"].values():
        block.update({
            "trials": 1, "time_nodes": 3, "grid_nx": 12, "grid_ny": 6,
            "cg_tol": 1e-5, "cg_maxiter": 120,
        })
    return cfg


def main():
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"v6 smoke requires CUDA; JAX backend is {jax.default_backend()}")
    output = SCRIPT_DIR / "outputs" / "prospective_v6_beta_ablation_smoke_v2"
    cfg = smoke_config()
    paths = v6_paths(output)
    paths["shared_results"].mkdir(parents=True, exist_ok=True)
    config_path = paths["shared_results"] / "smoke_config.json"
    if config_path.exists():
        existing = load_config(config_path)
        if existing != cfg:
            raise RuntimeError("existing v6 smoke output uses an incompatible config")
    else:
        write_json_atomic(config_path, cfg)
    result = run(cfg, output, "all")
    print(result["claims"])


if __name__ == "__main__":
    main()
