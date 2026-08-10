"""k-nearest-neighbour novelty in CLIP embedding space.

    NOV(b) = (1/k) * sum_{j in kNN(b)} (1 - <b, b_j>)

k=10 and the parent-sampling exponent alpha=4 are E&E's tuned values. All
vectors are L2-normalised, so cosine similarity is a plain dot product and the
whole thing is one matmul.

References are taken SEPARATELY rather than concatenated (archive UNION
rejects-ring, the classic Lehman-Stanley formulation): each reference block is
reduced to its k nearest distances first and the blocks are merged, rather
than copying the whole archive every generation.

Nothing in this module assumes a scale, and nothing downstream needs a
threshold in these units either: novelty RANKS entries for eviction rather
than gating admission, and a ranking is scale-free.
"""
from __future__ import annotations

import numpy as np


# Elements of the largest (queries x reference) block held at once. 16M
# float32 is 64 MB, which is what bounds a whole-archive rescore: the full
# matrix at the 20000 capacity would be 1.6 GB and is never materialised.
DISTANCE_BLOCK_ELEMS = 16_000_000


def knn_distances(queries: np.ndarray, reference: np.ndarray, k: int = 10,
                  exclude_self: bool = False,
                  block_elems: int = DISTANCE_BLOCK_ELEMS) -> np.ndarray:
    """(n, dim) x (m, dim) -> (n, k') cosine distances, sorted ascending.

    k' = min(k, m), or min(k, m-1) when exclude_self. An empty reference (or a
    single-entry one with exclude_self) yields (n, 0), which
    novelty_from_distances treats as 'no evidence'.

    exclude_self drops the single nearest match, which for a query drawn from
    the reference itself is the zero-distance self-match. Used when refreshing
    an archive entry's own novelty; without it every entry's nearest neighbour
    is itself and every novelty collapses toward zero.

    Computed in row BLOCKS. Only k distances per query survive, so the full
    (n, m) matrix is scratch - a whole-archive rescore at capacity would ask
    for over a GB of it at once. Blocking caps the scratch at block_elems and
    costs nothing, since the work is the same matmul either way.
    """
    q = np.asarray(queries, dtype=np.float32)
    q = q.reshape(len(q), -1) if len(q) else q.reshape(0, -1)
    r = np.asarray(reference, dtype=np.float32)
    n, m = len(q), len(r)
    if m == 0 or (exclude_self and m == 1):
        return np.zeros((n, 0), dtype=np.float32)

    want = int(min(k + (1 if exclude_self else 0), m))
    rows = max(1, int(block_elems) // m)
    out = np.empty((n, want), dtype=np.float32)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        dist = 1.0 - (q[start:stop] @ r.T)
        if want < m:
            idx = np.argpartition(dist, want - 1, axis=1)[:, :want]
            dist = np.take_along_axis(dist, idx, axis=1)
        out[start:stop] = np.sort(dist, axis=1)
    if exclude_self:
        out = out[:, 1:]
    return np.ascontiguousarray(out, dtype=np.float32)


def nearest_distance(queries: np.ndarray, reference: np.ndarray,
                     block_elems: int = DISTANCE_BLOCK_ELEMS) -> np.ndarray:
    """(n, dim) x (m, dim) -> (n,) cosine distance to the SINGLE closest row.

    The k=10 mean that novelty uses answers "how empty is this neighbourhood";
    this answers "is there already one of these", which is the question a
    separation rule asks. They are not interchangeable: ten neighbours at 0.05
    and one at 0.001 give a healthy-looking novelty of 0.045 for something the
    archive already holds.

    An empty reference gives inf - nothing is nearby because there is nothing.
    """
    q = np.asarray(queries, dtype=np.float32)
    q = q.reshape(len(q), -1) if len(q) else q.reshape(0, -1)
    r = np.asarray(reference, dtype=np.float32)
    n, m = len(q), len(r)
    if m == 0:
        return np.full(n, np.inf, dtype=np.float32)
    rows = max(1, int(block_elems) // m)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, rows):
        e = min(s + rows, n)
        out[s:e] = 1.0 - (q[s:e] @ r.T).max(axis=1)
    return out


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


def effective_sample_size(weights: np.ndarray, alpha: float = 1.0) -> float:
    """Kish ESS of p proportional to w^alpha: (sum w^a)^2 / sum w^(2a).

    How many entries are EFFECTIVELY in the running. N means uniform - the
    score is being ignored - and 1 means argmax. An all-zero weight vector is
    uniform by convention, matching sample_by_novelty's own fallback.

    Computed in log space. Naively, w**alpha overflows float64 well within the
    alpha range bisection explores - w=21 at alpha=256 is already 1e338 - and an
    inf makes the ratio NaN. Shifting by the maximum is exact here rather than
    approximate: the shift contributes e^M to the numerator squared and e^(2M)
    to the denominator, so it cancels identically.
    """
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    n = w.size
    if n == 0:
        return 0.0
    a = float(alpha)
    if a <= 0.0:
        return float(n)
    pos = w > 0.0
    if not pos.any():
        return float(n)
    ln = a * np.log(w[pos])
    u = np.exp(ln - ln.max())
    s1 = float(u.sum())
    s2 = float(np.dot(u, u))
    return float(s1 * s1 / s2) if s2 > 0.0 else float(n)


def alpha_for_ess(weights, target: float, lo: float = 0.0, hi: float = 512.0,
                  iters: int = 40) -> float:
    """The alpha whose ESS is `target`, by bisection on [lo, hi].

    Well posed because ESS is monotone NON-INCREASING in alpha for alpha >= 0.
    Writing S(t) = log sum_i exp(t * log w_i), we have
    log ESS(a) = 2 S(a) - S(2a), so d/da log ESS = 2[S'(a) - S'(2a)] <= 0:
    log-sum-exp is convex, so S' is non-decreasing. A monotone function has at
    most one crossing, so bisection cannot land on the wrong branch.
    """
    if effective_sample_size(weights, lo) <= target:
        return float(lo)
    if effective_sample_size(weights, hi) >= target:
        return float(hi)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if effective_sample_size(weights, mid) > target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# Bisection's upper bracket. 64 is not enough: a large archive whose scores
# barely separate needs a very sharp exponent to bring ESS down, and the
# bracket end binds silently. Safe at any size because ESS is computed in log
# space - see effective_sample_size.
ALPHA_CAP = 512.0


def banded_alpha(weights, alpha: float, ess_min: float, ess_max: float,
                 alpha_cap: float = ALPHA_CAP) -> float:
    """`alpha`, adjusted ONLY if its ESS falls outside [ess_min, ess_max].

    A band rather than a target: how concentrated a goal's matches are is real
    signal about the archive (see CLAUDE.md), and pinning ESS to one value
    would destroy it. The band exists only to stop the degenerate ends - too
    peaked and a repeated goal retraces one trajectory, too flat and the goal
    stops influencing the seed at all. Mirrors adaptive resampling in particle
    filters, which triggers on an ESS threshold rather than steering to one.

    The floor is capped at N/8: a floor is a demand for candidates that may not
    exist, and forcing most of a small archive into the pool makes the goal
    almost irrelevant.
    """
    n = int(np.asarray(weights).size)
    ess_min = min(float(ess_min), n / 8.0)
    e = effective_sample_size(weights, alpha)
    if e > float(ess_max):
        # ESS falls as alpha rises, so sharpen: search above the current alpha.
        return alpha_for_ess(weights, float(ess_max), lo=float(alpha),
                             hi=float(alpha_cap))
    if e < float(ess_min):
        return alpha_for_ess(weights, float(ess_min), lo=0.0, hi=float(alpha))
    return float(alpha)


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
