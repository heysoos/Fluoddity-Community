"""2-D PCA of the archive, for the map view.

Via the eigendecomposition of the dim x dim covariance rather than an SVD of the
data: at dim=512 that is a fixed 512x512 problem regardless of archive size, so
a refit costs the same at 200 entries as at 20,000.

This is a VIEW. Nothing in the search depends on it - novelty is always computed
in the full 512-d space.
"""
from __future__ import annotations

import numpy as np


class Projection:
    def __init__(self, n_components: int = 2):
        self.n_components = int(n_components)
        self.components: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self._prev: np.ndarray | None = None

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
        # eigh, not eig: the covariance is symmetric, so this is both faster and
        # guaranteed to return real orthonormal vectors.
        vals, vecs = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(vals)[::-1][: self.n_components]
        comp = np.ascontiguousarray(vecs[:, order].T, dtype=np.float32)

        # Eigenvectors have arbitrary sign. Aligning each new component to the
        # previous one is what stops the map mirroring itself on every refit.
        if self._prev is not None and self._prev.shape == comp.shape:
            flip = np.sign(np.sum(self._prev * comp, axis=1))
            flip[flip == 0] = 1.0
            comp = comp * flip[:, None]

        self.components = comp
        self.mean = mean.astype(np.float32)
        self._prev = comp.copy()
        return True

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        if not self.fitted:
            return np.zeros((x.shape[0], self.n_components), dtype=np.float32)
        return ((x - self.mean) @ self.components.T).astype(np.float32)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        self.fit(embeddings)
        return self.transform(embeddings)
