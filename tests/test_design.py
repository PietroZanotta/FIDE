import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mfsi.design import (
    OptimizerConfig,
    lexicographic_optimize,
    optimize_multistart_candidates,
    random_projective_starts,
)
from mfsi.measurements import GaussianSensor2D


def test_random_starts_are_separated():
    starts = random_projective_starts(jax.random.PRNGKey(0), 8, min_sep_rad=0.3)
    d = starts[:, 1] - starts[:, 0]
    sep = jnp.minimum(d, 2.0 * jnp.pi - d)
    assert jnp.all(sep >= 0.3)


def test_lexicographic_gradient_path():
    starts = jnp.asarray([[0.2, 1.0], [0.4, 1.2]], dtype=jnp.float64)
    cfg = OptimizerConfig(steps=80, learning_rate=0.03, constraint_penalty=1e5)

    population = lambda eta: jnp.sum((eta - jnp.asarray([0.3, 1.1])) ** 2) + 1.0
    risk = lambda eta: jnp.sum((eta - jnp.asarray([0.32, 1.08])) ** 2) + 2.0
    action = lambda eta: jnp.sum((eta - jnp.asarray([0.34, 1.06])) ** 2) + 3.0

    out = lexicographic_optimize(
        population,
        risk,
        action,
        starts,
        cfg,
        canonicalize=GaussianSensor2D.canonicalize,
    )
    assert out["law"].feasible
    assert out["conditioned"].feasible


def test_multistart_projects_every_adam_iterate():
    starts = jnp.asarray([[0.0]], dtype=jnp.float64)
    cfg = OptimizerConfig(steps=20, learning_rate=0.2)
    rows = optimize_multistart_candidates(
        lambda eta: -eta[0],
        starts,
        cfg,
        project_iterate=lambda eta: jnp.clip(eta, -0.1, 0.1),
        vectorize_starts=False,
    )
    assert jnp.isfinite(rows[-1].eta).all()
    assert float(rows[-1].eta[0]) <= 0.1


def test_batched_multistart_matches_serial_with_partial_final_batch():
    starts = jnp.asarray(
        [[-1.0, 0.2], [0.4, 1.2], [1.5, -0.3], [0.8, 0.9], [-0.2, -1.1]],
        dtype=jnp.float64,
    )
    cfg = OptimizerConfig(steps=20, learning_rate=0.03)
    objective = lambda eta: jnp.sum((eta - jnp.asarray([0.3, -0.4])) ** 2)
    serial = optimize_multistart_candidates(
        objective, starts, cfg, vectorize_starts=False
    )
    batched = optimize_multistart_candidates(
        objective, starts, cfg, vectorize_starts=True, start_batch_size=2
    )
    assert len(serial) == len(batched) == 2 * len(starts)
    for old, new in zip(serial, batched, strict=True):
        np.testing.assert_allclose(old.eta, new.eta, rtol=1e-12, atol=1e-12)
        assert new.value == pytest.approx(old.value, rel=1e-12, abs=1e-12)
        assert new.feasible == old.feasible
        np.testing.assert_allclose(new.violations, old.violations, rtol=0.0, atol=1e-12)


def test_multistart_batch_size_requires_vectorization():
    with pytest.raises(ValueError, match="requires vectorize_starts"):
        optimize_multistart_candidates(
            lambda eta: jnp.sum(eta**2),
            jnp.zeros((2, 1)),
            OptimizerConfig(steps=1),
            vectorize_starts=False,
            start_batch_size=2,
        )


def test_threaded_multistart_matches_serial_single_start_execution():
    starts = jnp.asarray(
        [[-1.0, 0.2], [0.4, 1.2], [1.5, -0.3], [0.8, 0.9]],
        dtype=jnp.float64,
    )
    cfg = OptimizerConfig(steps=20, learning_rate=0.03)
    objective = lambda eta: jnp.sum((eta - jnp.asarray([0.3, -0.4])) ** 2)
    serial = optimize_multistart_candidates(
        objective, starts, cfg, vectorize_starts=False
    )
    threaded = optimize_multistart_candidates(
        objective, starts, cfg, vectorize_starts=False, start_workers=2
    )
    assert len(serial) == len(threaded)
    for old, new in zip(serial, threaded, strict=True):
        np.testing.assert_array_equal(old.eta, new.eta)
        assert new.value == old.value
        assert new.feasible == old.feasible
        assert new.violations == old.violations
