from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mfsi.galerkin_tesseract import (
    assemble_galerkin_chunk_tesseract,
    assemble_galerkin_chunk_tesseract_forward,
    finalize_galerkin_statistics,
    is_tesseract_galerkin_available,
)

jax.config.update("jax_enable_x64", True)
pytestmark = pytest.mark.skipif(not is_tesseract_galerkin_available(), reason="native Galerkin Tesseract unavailable")


def _inputs(n=96, k=17, p=5, d=2):
    key = jax.random.PRNGKey(20260825)
    values = jax.random.normal(key, (n, k), dtype=jnp.float64)
    gradients = jax.random.normal(jax.random.fold_in(key, 1), (n, k, p, d), dtype=jnp.float64)
    weights = jax.nn.softmax(jax.random.normal(jax.random.fold_in(key, 2), (n,), dtype=jnp.float64))
    forcing = jax.random.normal(jax.random.fold_in(key, 3), (n,), dtype=jnp.float64)
    return values, gradients, weights, forcing


def _reference(values, gradients, weights, forcing):
    gram = jnp.einsum("n,njpd,nkpd->jk", weights, gradients, gradients)
    mean = jnp.einsum("n,nk->k", weights, values)
    forcing_sum = jnp.einsum("n,n->", weights, forcing)
    raw_load = jnp.einsum("n,n,nk->k", weights, forcing, values)
    return gram, raw_load, mean, forcing_sum


def test_native_chunk_matches_jax_and_is_deterministic():
    inputs = _inputs(); expected = _reference(*inputs)
    first = assemble_galerkin_chunk_tesseract_forward(*map(np.asarray, inputs))
    second = assemble_galerkin_chunk_tesseract_forward(*map(np.asarray, inputs))
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_allclose(first["gram"], expected[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(first["raw_load"], expected[1], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(first["basis_mean"], expected[2], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(first["forcing_sum"][0], expected[3], rtol=2e-13, atol=2e-13)


def test_tesseract_endpoint_matches_direct_native():
    inputs = _inputs(n=64, k=11, p=4)
    expected = assemble_galerkin_chunk_tesseract_forward(*map(np.asarray, inputs))
    actual = assemble_galerkin_chunk_tesseract(*inputs)
    for name in expected:
        np.testing.assert_allclose(np.asarray(actual[name]), expected[name], rtol=2e-15, atol=2e-15)


def test_additive_chunks_match_single_chunk_and_centering():
    inputs = _inputs(n=128, k=13, p=4); halves = []
    for begin, end in ((0, 53), (53, 128)):
        halves.append(assemble_galerkin_chunk_tesseract_forward(
            *[np.asarray(value[begin:end]) for value in inputs]))
    accumulated = {name: sum(row[name] for row in halves) for name in halves[0]}
    gram, load = finalize_galerkin_statistics(accumulated["gram"], accumulated["raw_load"],
                                               accumulated["basis_mean"], accumulated["forcing_sum"])
    expected = _reference(*inputs)
    np.testing.assert_allclose(gram, expected[0], rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(load, expected[1] - expected[3] * expected[2], rtol=3e-13, atol=3e-13)


def test_invalid_shape_and_negative_weight_fail_closed():
    values, gradients, weights, forcing = map(np.asarray, _inputs())
    with pytest.raises((TypeError, ValueError)):
        assemble_galerkin_chunk_tesseract_forward(values[:-1], gradients, weights, forcing)
    weights = weights.copy(); weights[0] = -1
    with pytest.raises(ValueError):
        assemble_galerkin_chunk_tesseract_forward(values, gradients, weights, forcing)
