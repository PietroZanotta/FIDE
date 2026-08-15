import jax
import jax.numpy as jnp

from mfsi.feasibility import prepare_polytope_2d, project_metric_polytope_2d


def test_metric_projection_square_and_gradient():
    A = jnp.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    b = jnp.ones(4)
    poly = prepare_polytope_2d(A, b)
    H = jnp.array([[2.,0.2],[0.2,1.]])
    beta = jnp.array([2.0, 0.3])
    out = project_metric_polytope_2d(beta, H, polytope=poly)
    assert jnp.all(A @ out.beta <= b + 1e-8)
    assert out.active

    f = lambda x: jnp.sum(project_metric_polytope_2d(jnp.array([x, 0.3]), H, polytope=poly).beta**2)
    g = jax.grad(f)(2.0)
    assert jnp.isfinite(g)
