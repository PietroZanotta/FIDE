import jax.numpy as jnp
import numpy as np

from mfsi.reference import (
    DomainPreservingReferenceFlow,
    MLPReferenceFlow,
    load_npz_checkpoint,
    save_npz_checkpoint,
)


def zero_velocity_params():
    return ({
        "W": jnp.zeros((7, 2), dtype=jnp.float64),
        "b": jnp.zeros((2,), dtype=jnp.float64),
    },)


def test_zero_velocity_rollout_is_constant():
    flow = MLPReferenceFlow(zero_velocity_params(), substeps_per_interval=3)
    x0 = jnp.asarray([[1.0, -2.0], [0.25, 0.5]], dtype=jnp.float64)
    times = jnp.asarray([0.0, 0.15, 0.7, 1.0], dtype=jnp.float64)
    nodes = flow.rollout(x0, times)
    assert nodes.shape == (4, 2, 2)
    assert jnp.allclose(nodes, jnp.broadcast_to(x0, nodes.shape))


def test_existing_npz_checkpoint_shape_is_preserved(tmp_path):
    path = tmp_path / "reference.npz"
    params = zero_velocity_params()
    save_npz_checkpoint(path, params, {"network": {"parameter_layers": 1}})
    loaded, metadata = load_npz_checkpoint(path)
    assert len(loaded) == 1
    assert np.asarray(loaded[0]["W"]).shape == (7, 2)
    assert metadata["network"]["parameter_layers"] == 1


def test_domain_preserving_map_roundtrip_and_interior():
    bounds = jnp.asarray([-650.0, 3000.0, -950.0, 1000.0])
    flow = DomainPreservingReferenceFlow(zero_velocity_params(), bounds, map_epsilon=1e-6)
    x = jnp.asarray([[-649.0, -949.0], [-76.0, -78.0], [2999.0, 999.0]])
    recovered = flow.to_physical(flow.to_latent(x))
    assert jnp.allclose(recovered, x, rtol=1e-11, atol=1e-10)
    boundary = jnp.asarray([[-650.0, -950.0], [3000.0, 1000.0]])
    mapped = flow.to_physical(flow.to_latent(boundary))
    assert jnp.all(mapped[:, 0] > bounds[0]) and jnp.all(mapped[:, 0] < bounds[1])
    assert jnp.all(mapped[:, 1] > bounds[2]) and jnp.all(mapped[:, 1] < bounds[3])


def test_domain_preserving_physical_velocity_matches_finite_difference():
    params = ({
        "W": jnp.zeros((7, 2), dtype=jnp.float64),
        "b": jnp.asarray([0.35, -0.2], dtype=jnp.float64),
    },)
    flow = DomainPreservingReferenceFlow(
        params, jnp.asarray([-650.0, 3000.0, -950.0, 1000.0]), map_epsilon=1e-6,
    )
    z = jnp.asarray([[0.2, -0.7], [-1.4, 1.1]], dtype=jnp.float64)
    t = jnp.asarray(0.37, dtype=jnp.float64)
    dt = 1.0e-6
    x = flow.to_physical(z)
    finite_difference = (flow.to_physical(z + dt * flow.latent_velocity(z, t)) - x) / dt
    expected = flow.physical_velocity_from_latent(z, t)
    assert jnp.allclose(finite_difference, expected, rtol=5e-7, atol=5e-7)


def test_domain_preserving_rollout_never_leaves_rectangle():
    params = ({
        "W": jnp.zeros((7, 2), dtype=jnp.float64),
        "b": jnp.asarray([50.0, -50.0], dtype=jnp.float64),
    },)
    bounds = jnp.asarray([-650.0, 3000.0, -950.0, 1000.0])
    flow = DomainPreservingReferenceFlow(params, bounds, map_epsilon=1e-6, substeps_per_interval=2)
    x0 = jnp.asarray([[-649.0, -949.0], [0.0, 0.0], [2999.0, 999.0]])
    nodes = flow.rollout(x0, jnp.linspace(0.0, 1.0, 11))
    assert jnp.all(nodes[..., 0] >= bounds[0]) and jnp.all(nodes[..., 0] <= bounds[1])
    assert jnp.all(nodes[..., 1] >= bounds[2]) and jnp.all(nodes[..., 1] <= bounds[3])
