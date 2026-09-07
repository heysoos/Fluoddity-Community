"""The projection asks for its top components rather than the whole spectrum.

A full eigh of a 512x512 covariance was 58 ms of the map's 113 ms PCA refit,
paid on the frame loop every 500 admissions; the Lanczos solver gives the
same vectors in 2 ms. Without scipy the dense path still fits.
"""
import sys

import numpy as np
import pytest

from services.archive_projection import Projection, top_eigenpairs


def _cloud(rng, n=400, d=64):
    x = rng.normal(size=(n, d))
    x[:, :6] *= np.array([9.0, 6.0, 4.0, 3.0, 2.5, 2.0])
    return x.astype(np.float32)


def _reference(x, k):
    c = x - x.mean(axis=0)
    cov = (c.T @ c) / (len(x) - 1)
    vals, vecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(vals)[::-1][:k]
    return vals[order], vecs[:, order].T


@pytest.mark.parametrize("k", [2, 8])
def test_the_solver_agrees_with_a_full_eigendecomposition(k):
    rng = np.random.default_rng(0)
    x = _cloud(rng)
    want_vals, want_vecs = _reference(x, k)
    c = x - x.mean(axis=0)
    cov = ((c.T @ c) / (len(x) - 1)).astype(np.float64)
    vals, vecs = top_eigenpairs(cov, k)
    assert vals.shape == (k,) and vecs.shape == (k, x.shape[1])
    assert np.allclose(vals, want_vals, rtol=1e-6)
    assert np.allclose(np.abs(np.sum(vecs * want_vecs, axis=1)), 1.0, atol=1e-6)


def test_the_projection_lands_where_it_always_did():
    rng = np.random.default_rng(1)
    x = _cloud(rng)
    want_vals, want_vecs = _reference(x, 2)
    p = Projection()
    assert p.fit(x)
    assert np.allclose(p.variances, want_vals, rtol=1e-5)
    assert np.allclose(np.abs(np.sum(p.components * want_vecs, axis=1)), 1.0,
                       atol=1e-5)


def test_without_scipy_the_dense_path_fits_the_same(monkeypatch):
    rng = np.random.default_rng(2)
    x = _cloud(rng, d=16)
    monkeypatch.setitem(sys.modules, "scipy.sparse.linalg", None)
    want_vals, want_vecs = _reference(x, 4)
    p = Projection(n_components=4)
    assert p.fit(x)
    assert np.allclose(p.variances, want_vals, rtol=1e-5)


def test_asking_for_nearly_every_component_uses_the_dense_path():
    """Lanczos needs k < d; a tiny space must not raise."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 4)).astype(np.float32)
    p = Projection(n_components=3)
    assert p.fit(x)
    assert p.components.shape == (3, 4)
