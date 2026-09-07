"""2-D PCA of the archive, for the map view.

Via the eigendecomposition of the dim x dim covariance rather than an SVD of the
data: at dim=512 that is a fixed 512x512 problem regardless of archive size, so
a refit costs the same at 200 entries as at 20,000.

This is a VIEW. Nothing in the search depends on it - novelty is always computed
in the full 512-d space.
"""
from __future__ import annotations

import numpy as np


def top_eigenpairs(cov: np.ndarray, k: int):
    """The k largest eigenpairs of a symmetric matrix, descending.

    -> (values (k,), vectors (k, d)). Lanczos through scipy when it is there
    and k leaves room for it; a dense eigh otherwise. Symmetric input is the
    caller's promise, which is what makes both paths return real orthonormal
    vectors.
    """
    d = cov.shape[0]
    k = int(k)
    if k < d - 1:
        try:
            from scipy.sparse.linalg import eigsh
        except Exception:               # noqa: BLE001 - optional dependency
            eigsh = None
        if eigsh is not None:
            try:
                # A fixed start vector keeps the fit deterministic.
                v0 = np.linspace(0.5, 1.5, d)
                vals, vecs = eigsh(cov, k=k, which="LA", v0=v0)
                order = np.argsort(vals)[::-1]
                return vals[order], vecs[:, order].T
            except Exception:           # noqa: BLE001 - fall through to dense
                pass
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:k]
    return vals[order], vecs[:, order].T


class Projection:
    def __init__(self, n_components: int = 2):
        self.n_components = int(n_components)
        self.components: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        # The retained eigenvalues, descending - the variance along each
        # component. The map does not need them; the search does, because
        # whitening is what makes an extrapolation distance mean the same thing
        # along a dominant axis and a minor one. See goal_source.latent_goal.
        self.variances: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        # Bumped on every successful fit, so a view can cache transform()'s
        # output - an (n, dim) @ (dim, 2) matmul - instead of redoing it per
        # frame. See ArchiveWindowMixin._map_points.
        self.version = 0

    @property
    def fitted(self) -> bool:
        return self.components is not None

    def fit(self, embeddings: np.ndarray) -> bool:
        """-> did it fit? False when there is not enough data yet."""
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] <= self.n_components:
            return False

        mean = x.mean(axis=0)
        centred = x - mean
        cov = (centred.T @ centred) / max(1, x.shape[0] - 1)
        top, comp = top_eigenpairs(cov.astype(np.float64), self.n_components)
        comp = np.ascontiguousarray(comp, dtype=np.float32)

        # Eigenvectors have arbitrary sign. Aligning each new component to the
        # previous one is what stops the map mirroring itself on every refit.
        if self._prev is not None and self._prev.shape == comp.shape:
            flip = np.sign(np.sum(self._prev * comp, axis=1))
            flip[flip == 0] = 1.0
            comp = comp * flip[:, None]

        self.components = comp
        self.mean = mean.astype(np.float32)
        # Clamped at 0: eigh can return a tiny negative for a near-zero
        # eigenvalue, and a negative variance becomes a NaN standard deviation.
        self.variances = np.maximum(top, 0.0).astype(np.float32)
        self._prev = comp.copy()
        self.version += 1
        return True

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        if not self.fitted:
            return np.zeros((x.shape[0], self.n_components), dtype=np.float32)
        return ((x - self.mean) @ self.components.T).astype(np.float32)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        self.fit(embeddings)
        return self.transform(embeddings)
