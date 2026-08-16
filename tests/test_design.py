import jax
import jax.numpy as jnp

from mfsi.design import OptimizerConfig, lexicographic_optimize, random_projective_starts
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
