import jax.numpy as jnp

from mfsi.moments import AnchoredGLSConfig, fit_anchored_basis_gls


def test_generic_anchored_basis_keeps_endpoints():
    basis = lambda t: jnp.stack([t*(1-t), t*(1-t)*(2*t-1)], axis=-1)
    basis_d = lambda t: jnp.stack([1-2*t, -6*t*t+6*t-1], axis=-1)
    t = jnp.array([0., .25, .5, .75, 1.])
    c0 = jnp.array([0.2, 0.1])
    c1 = jnp.array([0.4, 0.3])
    coef = jnp.array([[0.1,-0.05],[0.02,0.03]])
    y = (1-t[:,None])*c0 + t[:,None]*c1 + basis(t)@coef
    V = jnp.tile(jnp.eye(2)[None,:,:]*1e-3, (len(t),1,1))
    fit = fit_anchored_basis_gls(t, y, V, c0, c1, t, basis=basis, basis_derivative=basis_d, cfg=AnchoredGLSConfig())
    assert jnp.allclose(fit.c[0], c0, atol=1e-8)
    assert jnp.allclose(fit.c[-1], c1, atol=1e-8)
