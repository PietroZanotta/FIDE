"""Fast compatibility/unit checks for the additive core extensions.

This does not replace the repository's existing toy smoke. It is intentionally free
of the experiment projection/reference dependencies so it can isolate the six modified
core files before any expensive run.
"""
from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
jax.config.update("jax_enable_x64", True)

from mfsi.design import point_box_violation, point_separation_violation, random_point_sensor_starts
from mfsi.grid import CartesianGrid2D, RectangularGrid2D
from mfsi.measurements import GaussianPointSensors2D, GaussianSensor2D
from mfsi.metrics import gaussian_mmd2_grid_mass, gaussian_mmd_kernel_rect
from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from mfsi.raster import RasterConfig, rasterize_projected_particles_rect
from mfsi.projection import EmpiricalIProjector, IProjectionConfig


def main() -> None:
    # Historical APIs still behave as before at the interface level.
    old_grid = CartesianGrid2D(half_width=3.2, n=51)
    assert old_grid.points().shape == (51, 51, 2)
    old_family = GaussianSensor2D()
    assert old_family.features(jnp.zeros((3, 2)), jnp.asarray([0.2, 1.1])).shape == (3, 2)

    grid = RectangularGrid2D(0.0, 2.0, 0.0, 1.0, 128, 64)
    assert grid.shape == (64, 128)
    grid.require_isotropic_spacing()

    family = GaussianPointSensors2D(width=0.12, n_sensors=4)
    eta = jnp.asarray([0.3, 0.3, 0.7, 0.7, 1.3, 0.3, 1.7, 0.7])
    x = jnp.asarray([[0.5, 0.5], [1.0, 0.5]])
    grad = family.feature_gradients(x, eta)
    jac = jax.jacfwd(lambda xx: family.features(xx[None, :], eta)[0])(x[0])
    np.testing.assert_allclose(np.asarray(jac), np.asarray(grad[0]), rtol=1e-8, atol=1e-8)

    box = point_box_violation(n_sensors=4, x_bounds=(0.24, 1.76), y_bounds=(0.24, 0.76))
    sep = point_separation_violation(0.24, n_sensors=4)
    starts = random_point_sensor_starts(
        jax.random.PRNGKey(0), 8, n_sensors=4,
        x_bounds=(0.24, 1.76), y_bounds=(0.24, 0.76), min_sep=0.24,
    )
    assert all(float(box(s)) <= 1e-12 and float(sep(s)) <= 1e-12 for s in starts)

    t_obs = jnp.linspace(0.0, 1.0, 9)
    y = jnp.stack([0.2 + 0.3 * t_obs + 0.1 * jnp.sin(2 * jnp.pi * t_obs),
                   0.4 + 0.05 * jnp.cos(jnp.pi * t_obs)], axis=1)
    t0, h = 0.43, 1e-5
    spline = AnchoredCubicSplineReconstructor(
        t_obs, jnp.asarray([t0 - h, t0, t0 + h]),
        AnchoredCubicSplineConfig(internal_knots=3, smoothing=1e-5),
    )
    fit = spline.reconstruct(y, y[0], y[-1])
    fd = (fit.c[2] - fit.c[0]) / (2 * h)
    np.testing.assert_allclose(np.asarray(fd), np.asarray(fit.c_dot[1]), rtol=2e-5, atol=2e-6)
    g = jax.grad(lambda yy: jnp.sum(spline.reconstruct(yy, yy[0], yy[-1]).c ** 2))(y)
    assert np.all(np.isfinite(np.asarray(g)))

    # Existing empirical I-projection is dimension-generic and its custom VJP must
    # agree with a centered finite difference at an interior four-moment target.
    key = jax.random.PRNGKey(17)
    phi = 0.4 * jax.random.normal(key, (96, 4), dtype=jnp.float64)
    base = jnp.linspace(1.0, 2.0, phi.shape[0], dtype=jnp.float64)
    base = base / jnp.sum(base)
    lam_true = jnp.asarray([0.18, -0.12, 0.09, 0.05], dtype=jnp.float64)
    tilted = jax.nn.softmax(jnp.log(base) + phi @ lam_true)
    target = tilted @ phi
    projector = EmpiricalIProjector(IProjectionConfig(
        max_steps=100, residual_tol=1e-12, newton_ridge=1e-10, implicit_ridge=0.0
    ))
    state = projector.project(phi, base, target)
    assert float(jnp.linalg.norm(state.residual)) < 1e-9
    objective = lambda c: jnp.sum(projector.project(phi, base, c).lam ** 2)
    ad = jax.grad(objective)(target)
    dh = 1e-6
    ep = target.at[0].add(dh)
    em = target.at[0].add(-dh)
    fd0 = (objective(ep) - objective(em)) / (2.0 * dh)
    np.testing.assert_allclose(float(fd0), float(ad[0]), rtol=2e-4, atol=2e-6)

    points = jnp.asarray([[0.1, 0.1], [0.1, 0.1], [1.9, 0.9], [1.0, 0.5]])
    weights = jnp.asarray([0.2, 0.3, 0.1, 0.4])
    forcing = jnp.asarray([1.0, -1.0, 2.0, 0.0])
    ras = rasterize_projected_particles_rect(points, weights, forcing, grid, RasterConfig())
    np.testing.assert_allclose(float(jnp.sum(ras.mass)), 1.0, atol=1e-12)
    np.testing.assert_allclose(float(jnp.sum(ras.mass * ras.h)), 0.0, atol=1e-12)

    kernel = gaussian_mmd_kernel_rect(grid.nx, grid.ny, grid.dx, grid.dy, 0.1)
    p = jnp.zeros(grid.shape).at[10, 10].set(1.0)
    assert float(gaussian_mmd2_grid_mass(p, p, kernel)) < 1e-12
    print("vortex additive-core verification: PASS")


if __name__ == "__main__":
    main()
