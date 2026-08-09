"""Bounded training orchestration used by smoke and acceptance runs."""

from __future__ import annotations

import numpy as np

from .homometric import PopulationSupport
from .network import PriorParameters
from .training import (
    fine_tune_conditional,
    fit_prior_mle,
    generate_training_samples,
    make_conditional_tasks,
)


def train_variants(
    config: dict,
    support: PopulationSupport,
    true_params: PriorParameters,
    initial_params: PriorParameters,
    rng: np.random.Generator,
) -> dict:
    training = config["training"]
    prior_ids = generate_training_samples(
        true_params, support, int(training["prior_samples"]), rng
    )
    mle, mle_trace = fit_prior_mle(
        initial_params,
        support,
        prior_ids,
        steps=int(training["mle_steps"]),
        learning_rate=float(training["learning_rate"]),
    )
    tasks = make_conditional_tasks(
        true_params,
        support,
        training["task_tilts"],
        int(training["samples_per_task"]),
        rng,
    )
    stopgrad, stopgrad_trace = fine_tune_conditional(
        mle,
        support,
        tasks,
        steps=int(training["conditional_steps"]),
        learning_rate=float(training["learning_rate"]),
        ess_weight=float(training["ess_weight"]),
        differentiate_dual=False,
    )
    full, full_trace = fine_tune_conditional(
        mle,
        support,
        tasks,
        steps=int(training["conditional_steps"]),
        learning_rate=float(training["learning_rate"]),
        ess_weight=float(training["ess_weight"]),
        differentiate_dual=True,
    )
    return {
        "mle": mle,
        "stopgrad": stopgrad,
        "full_e2e": full,
        "tasks": tasks,
        "traces": {
            "mle": mle_trace,
            "stopgrad": stopgrad_trace,
            "full_e2e": full_trace,
        },
    }
