"""Independent reduced Full-E2E directional derivative acceptance job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from manybody_completion.energy import PhysicalParameters  # noqa: E402
from manybody_completion.homometric import motif_coordinates  # noqa: E402
from manybody_completion.network import (  # noqa: E402
    FlowNetworkConfig,
    initialize_flow_network,
)
from manybody_completion.observables import PairBasis, ensemble_pair_moments  # noqa: E402
from manybody_completion.routing import AblationMode  # noqa: E402
from manybody_completion.solvers import (  # noqa: E402
    LocalJaxBackend,
    ProjectionOptions,
    RelaxationOptions,
)
from manybody_completion.flow import SamplingOptions  # noqa: E402
from manybody_completion.training import FineTuneWeights, route_objective  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "composed_gradient_probe.json",
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64
    box = jnp.ones(2, dtype=dtype)
    basis = PairBasis.uniform(1, 0.30, 0.30, 0.08, dtype=dtype)
    target = motif_coordinates(0, box, dtype)[None, None, ...]
    condition = jnp.zeros((1, 1), dtype=dtype)
    target_moments = ensemble_pair_moments(target[0], box, basis)[None, :]
    config = FlowNetworkConfig(
        hidden_dim=4,
        message_dim=4,
        num_layers=1,
        radial_basis_size=3,
        radial_max=0.5,
        radial_width=0.1,
        time_frequencies=1,
        velocity_scale=1.0,
    )
    parameters = initialize_flow_network(
        jax.random.PRNGKey(0), 1, config, dtype=dtype
    )
    backend = LocalJaxBackend(
        box=box,
        basis=basis,
        moment_scales=jnp.ones(1, dtype=dtype),
        physical=PhysicalParameters(r0=0.22, kappa=20.0),
        relaxation_options=RelaxationOptions(
            num_steps=1, step_size=0.05, tolerance=1.0
        ),
        projection_options=ProjectionOptions(
            num_steps=1, tolerance=1.0, ridge=1e-5
        ),
    )
    fixed_key = jax.random.PRNGKey(1)
    loss = lambda model: route_objective(
        model,
        AblationMode.FULL_E2E,
        target,
        condition,
        target_moments,
        fixed_key,
        backend,
        config,
        SamplingOptions(num_steps=1, method="euler"),
        FineTuneWeights(
            flow_matching=1.0,
            observed=1.0,
            physical=1.0,
            correction=1.0,
        ),
    )[0]
    leaves, structure = jax.tree_util.tree_flatten(parameters)
    keys = jax.random.split(jax.random.PRNGKey(2), len(leaves))
    direction_leaves = [
        jax.random.normal(key, leaf.shape, leaf.dtype)
        for key, leaf in zip(keys, leaves)
    ]
    norm = jnp.sqrt(sum(jnp.sum(value * value) for value in direction_leaves))
    direction = jax.tree_util.tree_unflatten(
        structure, [value / norm for value in direction_leaves]
    )
    primal, autodiff = jax.jvp(loss, (parameters,), (direction,))

    def shift(scale: float):
        return jax.tree_util.tree_map(
            lambda parameter, delta: parameter + scale * delta,
            parameters,
            direction,
        )

    epsilons = [0.003, 0.001, 0.0003]
    finite = [
        float((loss(shift(epsilon)) - loss(shift(-epsilon))) / (2.0 * epsilon))
        for epsilon in epsilons
    ]
    autodiff_value = float(autodiff)
    relative = [
        abs(value - autodiff_value) / max(abs(autodiff_value), 1e-12)
        for value in finite
    ]
    report = {
        "status": "passed" if np.isfinite(relative).all() and min(relative) < 1e-4 else "failed",
        "scope": "reduced Full-E2E implementation probe",
        "loss": float(primal),
        "autodiff_directional_derivative": autodiff_value,
        "epsilons": epsilons,
        "finite_differences": finite,
        "relative_errors": relative,
        "best_relative_error": min(relative),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.experiment_report is not None:
        experiment = json.loads(args.experiment_report.read_text(encoding="utf-8"))
        experiment["gradient_check"] = report
        args.experiment_report.write_text(
            json.dumps(experiment, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
