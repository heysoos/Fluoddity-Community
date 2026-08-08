"""k-nearest-neighbour novelty in CLIP embedding space.

    NOV(b) = (1/k) * sum_{j in kNN(b)} (1 - <b, b_j>)

k=10 and the parent-sampling exponent alpha=4 are E&E's tuned values. All
vectors are L2-normalised, so cosine similarity is a plain dot product and the
whole thing is one matmul.

References are taken SEPARATELY rather than concatenated. Novelty is measured
against archive UNION rejects-ring (the classic Lehman-Stanley formulation:
archive plus current population), and gluing a 2k ring onto a 20k archive every
generation would copy 40 MB - more than the novelty computation costs. Instead
each reference block is reduced to its k nearest distances and the blocks are
merged.

Scale note, measured 2026-08-07: over 97 viable presets the mean pairwise
cosine distance is 0.159 with std 0.063, so novelty here lives in a compressed
range. Nothing in this module assumes a scale - the adaptive threshold in
services/archive.py is what adapts to it.
"""
from __future__ import annotations

import numpy as np


def knn_distances(queries: np.ndarray, reference: np.ndarray, k: int = 10,
                  exclude_self: bool = False) -> np.ndarray:
    """(n, dim) x (m, dim) -> (n, k') cosine distances, sorted ascending.

    k' = min(k, m), or min(k, m-1) when exclude_self. An empty reference (or a
    single-entry one with exclude_self) yields (n, 0), which
    novelty_from_distances treats as 'no evidence'.

    exclude_self drops the single nearest match, which for a query drawn from
    the reference itself is the zero-distance self-match. Used when refreshing
    an archive entry's own novelty; without it every entry's nearest neighbour
    is itself and every novelty collapses toward zero.
    """
    q = np.asarray(queries, dtype=np.float32)
    q = q.reshape(len(q), -1) if len(q) else q.reshape(0, -1)
    r = np.asarray(reference, dtype=np.float32)
    n, m = len(q), len(r)
    if m == 0 or (exclude_self and m == 1):
        return np.zeros((n, 0), dtype=np.float32)

    want = int(min(k + (1 if exclude_self else 0), m))
    dist = 1.0 - (q @ r.T)
    if want < m:
        idx = np.argpartition(dist, want - 1, axis=1)[:, :want]
        d = np.take_along_axis(dist, idx, axis=1)
    else:
        d = dist
    d = np.sort(d, axis=1)
    if exclude_self:
        d = d[:, 1:]
    return np.ascontiguousarray(d, dtype=np.float32)


def novelty_from_distances(dists: list[np.ndarray], k: int = 10) -> np.ndarray:
    """Merge per-reference distance blocks into one novelty value per query.

    With no reference at all every query is maximally novel by convention (1.0),
    which is what makes a cold archive bootstrap instead of dividing by zero.
    """
    blocks = [np.asarray(d, dtype=np.float32) for d in dists]
    usable = [b for b in blocks if b.ndim == 2 and b.shape[1] > 0]
    if not usable:
        n = int(blocks[0].shape[0]) if blocks else 0
        return np.ones(n, dtype=np.float32)
    d = np.concatenate(usable, axis=1)
    d = np.sort(d, axis=1)[:, : int(min(k, d.shape[1]))]
    return d.mean(axis=1).astype(np.float32)


def knn_novelty(queries: np.ndarray, reference: np.ndarray, k: int = 10,
                exclude_self: bool = False) -> np.ndarray:
    """Single-reference convenience wrapper."""
    return novelty_from_distances(
        [knn_distances(queries, reference, k, exclude_self)], k
    )


def sample_by_novelty(novelty: np.ndarray, n: int, rng, alpha: float = 4.0):
    """Indices sampled with p proportional to NOV^alpha, with replacement.

    alpha=0 is uniform - E&E's 'Random GA' baseline, reachable from the UI
    without a separate code path. Negative novelty cannot occur for unit
    vectors but is clamped anyway so a NaN-repaired value cannot flip a weight
    negative and poison the distribution.
    """
    v = np.asarray(novelty, dtype=np.float64).reshape(-1)
    m = v.size
    if m == 0:
        raise ValueError("cannot sample a parent from an empty archive")
    w = np.clip(v, 0.0, None) ** float(alpha)
    total = float(w.sum())
    p = (w / total) if (total > 0.0 and np.isfinite(total)) else np.full(m, 1.0 / m)
    return rng.choice(m, size=int(n), replace=True, p=p)


class RejectsRing:
    """The most recent rejected descriptors, for novelty only.

    Without this, novelty is measured against a filtered history: a gated
    archive has no memory of the regions it just rejected, so the search
    re-explores them forever. Never persisted - it is about the last few
    minutes, not the run.
    """

    def __init__(self, capacity: int = 2048, dim: int = 512):
        self.capacity = int(capacity)
        self._buf = np.zeros((self.capacity, int(dim)), dtype=np.float32)
        self._n = 0
        self._pos = 0

    def __len__(self) -> int:
        return self._n

    def add(self, embeddings: np.ndarray) -> None:
        e = np.asarray(embeddings, dtype=np.float32).reshape(-1, self._buf.shape[1])
        for row in e:
            self._buf[self._pos] = row
            self._pos = (self._pos + 1) % self.capacity
            self._n = min(self._n + 1, self.capacity)

    def view(self) -> np.ndarray:
        return self._buf[: self._n]

    def clear(self) -> None:
        self._n = 0
        self._pos = 0
