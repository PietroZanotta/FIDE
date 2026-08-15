import jax.numpy as jnp
import numpy as np

from mfsi.reference import MLPReferenceFlow, load_npz_checkpoint, save_npz_checkpoint


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
