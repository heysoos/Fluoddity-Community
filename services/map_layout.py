"""The archive map's 2-D layout: PCA, UMAP, and what keeps a refit stable.

PCA holds 46% of the variance at every archive size, which through two linear
axes of a CLIP space is a blob - it crowds, but more importantly it does not
separate. UMAP is the alternative, and it is optional: absent, the map is
exactly what it always was.

UMAP lays out a SNAPSHOT. Entries admitted afterwards are placed by kNN
interpolation rather than by umap.transform, so the archive cache holds only
numbers - there is no reducer pickle to break across a library upgrade - and
placement degrades smoothly instead of failing.

This is a VIEW. The search holds its own Projection(LATENT_DIMS) for whitening
and nothing here reaches it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from services.archive_projection import Projection
from services.novelty import knn_distances

# UMAP fits on this many PCA components rather than raw 512-d. The reduction is
# an eigendecomposition of the dim x dim covariance, so it costs the same at
# 200 entries as at 20,000.
REDUCE_DIMS = 40
# Growth past the size a layout was fitted at before it refits itself.
STALE_AT = 0.25
PLACE_K = 8

ENGINES = ("pca", "umap")
ENGINE_LABELS = {"pca": "PCA", "umap": "UMAP"}


def staleness(size: int, fitted_n: int) -> float:
    """How far the archive has grown past the fit, as a fraction of it.

    Shrinking is not staleness: eviction removes the least novel, which does
    not move anything that is left.
    """
    if fitted_n <= 0:
        return 1.0
    return max(0.0, (float(size) - float(fitted_n)) / float(fitted_n))


def knn_place(queries: np.ndarray, reference: np.ndarray, positions: np.ndarray,
              k: int = PLACE_K) -> np.ndarray:
    """Place `queries` at the mean position of their nearest `reference` rows.

    How an entry admitted after the fit gets onto the map. An empty reference
    gives the origin, which is what an archive with no layout yet should draw.
    """
    q = np.asarray(queries, dtype=np.float32).reshape(len(queries), -1)
    r = np.asarray(reference, dtype=np.float32)
    pos = np.asarray(positions, dtype=np.float32)
    if len(q) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(r) == 0 or len(pos) == 0:
        return np.zeros((len(q), 2), dtype=np.float32)

    k = int(max(1, min(k, len(r))))
    # Cosine distance and the index that produced it. knn_distances returns
    # only the distances, so the argsort is redone here over the same matmul.
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    rn = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
    out = np.zeros((len(q), 2), dtype=np.float32)
    block = max(1, 4_000_000 // max(1, len(r)))
    for lo in range(0, len(q), block):
        sims = qn[lo:lo + block] @ rn.T
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out[lo:lo + block] = pos[idx].mean(axis=1)
    return out.astype(np.float32)


def procrustes_align(moving: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate/reflect+translate `moving` onto `target`, without scaling.

    UMAP can rotate AND mirror between fits, and a mirrored map is a new map.
    Scale is left alone because the view normalises it anyway.
    """
    a = np.asarray(moving, dtype=np.float32)
    b = np.asarray(target, dtype=np.float32)
    n = min(len(a), len(b))
    if n < 2:
        return a
    ac, bc = a[:n].mean(axis=0), b[:n].mean(axis=0)
    u, _s, vt = np.linalg.svd((a[:n] - ac).T @ (b[:n] - bc))
    rot = (u @ vt).T
    return ((a - ac) @ rot.T + bc).astype(np.float32)


@dataclass
class LayoutCache:
    """A fitted layout, as it sits in `<archive>/map_layout.npz`."""
    pos: np.ndarray
    keys: list                      # (layout signature, entry id)
    encoder: str
    reduction: np.ndarray
    reduction_mean: np.ndarray
    fitted_n: int
    engine: str

    def save(self, path) -> None:
        sigs = np.array([str(s) for s, _ in self.keys], dtype=object)
        ids = np.array([int(i) for _, i in self.keys], dtype=np.int64)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(Path(path), pos=self.pos, sigs=sigs, ids=ids,
                 encoder=np.array(self.encoder), reduction=self.reduction,
                 reduction_mean=self.reduction_mean,
                 fitted_n=np.array(int(self.fitted_n)),
                 engine=np.array(self.engine), allow_pickle=True)

    @classmethod
    def load(cls, path, encoder: str) -> "LayoutCache | None":
        """-> the cache, or None for anything at all wrong with it.

        A view, so the fallback is a refit. A cache from another encoder is
        refused rather than trusted: at equal width a foreign vector is
        silently wrong, which is why load_from_store already refuses one.
        """
        p = Path(path)
        if not p.exists():
            return None
        try:
            with np.load(p, allow_pickle=True) as z:
                got = str(z["encoder"])
                if got != str(encoder):
                    return None
                keys = [(str(s), int(i))
                        for s, i in zip(z["sigs"], z["ids"])]
                return cls(pos=np.asarray(z["pos"], dtype=np.float32),
                           keys=keys, encoder=got,
                           reduction=np.asarray(z["reduction"], np.float32),
                           reduction_mean=np.asarray(z["reduction_mean"],
                                                     np.float32),
                           fitted_n=int(z["fitted_n"]),
                           engine=str(z["engine"]))
        except Exception:
            return None

    def lookup(self, keys) -> tuple[np.ndarray, np.ndarray]:
        """-> (positions, hit mask) for `keys`, in order.

        By (layout signature, id): an id is unique only inside one brain's
        directory and the map pools every layout, so a bare id would hand back
        another brain's position.
        """
        where = {k: n for n, k in enumerate(self.keys)}
        pos = np.zeros((len(keys), 2), dtype=np.float32)
        hit = np.zeros(len(keys), dtype=bool)
        for n, k in enumerate(keys):
            j = where.get((str(k[0]), int(k[1])))
            if j is not None and j < len(self.pos):
                pos[n], hit[n] = self.pos[j], True
        return pos, hit


class _BaseLayout:
    engine = "pca"

    def __init__(self):
        self.version = 0
        self.fitted_n = 0
        self.unavailable = ""

    @property
    def fitted(self) -> bool:
        raise NotImplementedError


class PcaLayout(_BaseLayout):
    """The historical map, unchanged. Also the fallback for everything."""

    engine = "pca"

    def __init__(self):
        super().__init__()
        self._proj = Projection(2)

    @property
    def fitted(self) -> bool:
        return self._proj.fitted

    def fit(self, embeddings, keys=None) -> bool:
        if not self._proj.fit(embeddings):
            return False
        self.fitted_n = len(embeddings)
        self.version += 1
        return True

    def transform(self, embeddings, keys=None) -> np.ndarray:
        return self._proj.transform(embeddings)


class UmapLayout(_BaseLayout):
    """A UMAP of the archive, fitted on a PCA pre-reduction.

    `fit` is slow and belongs on a worker thread; nothing here touches the
    archive, so it only ever sees the copy it was handed.
    """

    engine = "umap"

    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1,
                 seed: int = 42):
        super().__init__()
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.seed = int(seed)
        self._reduce = Projection(REDUCE_DIMS)
        self._ref: np.ndarray | None = None      # reduced coords of the fit
        self._pos: np.ndarray | None = None      # their 2-D positions
        self._keys: list = []

    @property
    def fitted(self) -> bool:
        return self._pos is not None

    @staticmethod
    def available() -> bool:
        try:
            import umap  # noqa: F401
        except Exception:
            return False
        return True

    def fit(self, embeddings, keys=None, init_pos=None) -> bool:
        """-> did it fit? False when umap is absent or there is too little data.

        `init_pos` seeds the optimisation from the previous layout, and the
        result is aligned back onto it: without both, a refit rearranges the
        map the user had already learned.
        """
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or len(x) <= max(3, self.n_neighbors):
            return False
        try:
            import umap
        except Exception as exc:
            self.unavailable = str(exc) or "umap-learn is not installed"
            return False

        self._reduce.n_components = min(REDUCE_DIMS, x.shape[1], len(x) - 1)
        if not self._reduce.fit(x):
            return False
        red = self._reduce.transform(x)

        kw = dict(n_components=2, n_neighbors=min(self.n_neighbors, len(x) - 1),
                  min_dist=self.min_dist, metric="cosine",
                  random_state=self.seed, verbose=False)
        if init_pos is not None and len(init_pos) == len(x):
            kw["init"] = np.asarray(init_pos, dtype=np.float32)
        try:
            pos = umap.UMAP(**kw).fit_transform(red)
        except Exception as exc:
            self.unavailable = str(exc)
            return False

        pos = np.asarray(pos, dtype=np.float32)
        if init_pos is not None and len(init_pos) == len(pos):
            pos = procrustes_align(pos, np.asarray(init_pos, np.float32))

        self._ref, self._pos = red, pos
        self._keys = list(keys or [])
        self.fitted_n = len(x)
        self.unavailable = ""
        self.version += 1
        return True

    def adopt(self, cache: LayoutCache, embeddings, keys) -> bool:
        """Take a cached layout as the fit, without refitting.

        The cache holds positions, not a reducer, so the reference coordinates
        are rebuilt by re-reducing the embeddings the cached keys point at.
        """
        pos, hit = cache.lookup(keys)
        if not hit.any():
            return False
        x = np.asarray(embeddings, dtype=np.float32)[hit]
        self._reduce.components = cache.reduction
        self._reduce.mean = cache.reduction_mean
        self._reduce.n_components = len(cache.reduction)
        self._ref = self._reduce.transform(x)
        self._pos = pos[hit]
        self._keys = [k for k, h in zip(keys, hit) if h]
        self.fitted_n = int(cache.fitted_n)
        self.version += 1
        return True

    def transform(self, embeddings, keys=None) -> np.ndarray:
        """Fitted rows keep their position; the rest are placed by kNN."""
        x = np.asarray(embeddings, dtype=np.float32)
        if not self.fitted:
            return np.zeros((len(x), 2), dtype=np.float32)
        red = self._reduce.transform(x)
        out = knn_place(red, self._ref, self._pos)
        if keys:
            where = {k: n for n, k in enumerate(self._keys)}
            for n, k in enumerate(keys):
                j = where.get((str(k[0]), int(k[1])))
                if j is not None:
                    out[n] = self._pos[j]
        return out

    def to_cache(self, encoder: str) -> LayoutCache | None:
        if not self.fitted:
            return None
        return LayoutCache(
            pos=self._pos, keys=list(self._keys), encoder=str(encoder),
            reduction=np.asarray(self._reduce.components, np.float32),
            reduction_mean=np.asarray(self._reduce.mean, np.float32),
            fitted_n=int(self.fitted_n), engine=self.engine)


def make_layout(engine: str):
    return UmapLayout() if str(engine) == "umap" else PcaLayout()
