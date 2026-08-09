import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.energies import EnergyParameters, total_energy_per_configuration
from manybody_completion.simulators import LangevinConfig, simulate_overdamped_langevin


def test_zero_temperature_pair_simulation_reduces_energy():
    box = jnp.array([1.0, 1.0])
    x0 = jnp.array([[[0.50, 0.50], [0.53, 0.50]]], dtype=jnp.float64)
    params = EnergyParameters(r0=0.16, kappa=30.0)
    config = LangevinConfig(
        num_steps=100,
        time_step=1e-3,
        temperature=0.0,
        max_drift_norm=0.02,
    )
    before = total_energy_per_configuration(x0, box, params, "pair")
    xr, _ = simulate_overdamped_langevin(jax.random.PRNGKey(0), x0, box, params, config, "pair")
    after = total_energy_per_configuration(xr, box, params, "pair")
    assert float(after[0]) < float(before[0])
    assert np.isfinite(np.asarray(xr)).all()


def test_simulator_is_seed_reproducible():
    box = jnp.array([1.0, 1.0])
    x0 = jnp.array([[[0.1, 0.2], [0.4, 0.7], [0.8, 0.9]]], dtype=jnp.float64)
    params = EnergyParameters()
    config = LangevinConfig(num_steps=5, temperature=0.01)
    key = jax.random.PRNGKey(9)
    x1, _ = simulate_overdamped_langevin(key, x0, box, params, config, "pair")
    x2, _ = simulate_overdamped_langevin(key, x0, box, params, config, "pair")
    np.testing.assert_array_equal(x1, x2)
