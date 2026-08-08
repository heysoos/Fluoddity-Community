"""The exploration archive: admission gates, capacity, and novelty bookkeeping.

Three gates, all of which must pass:
  viable   - the tile is not black or blown out
  alive    - liveness >= liveness_min (ASAL Eq.3; a frozen canvas scores ~0)
  novel    - kNN novelty >= an ADAPTIVE threshold

The threshold adapts because a fixed one either floods a rich region or starves
a barren one, and which happens depends on the preset the user loaded. Measured
2026-08-07, this is load-bearing rather than a convenience: over 97 viable
presets the mean pairwise cosine distance is 0.159, so novelty lives in a
compressed range and a threshold guessed from a wider sample would be ~20% too
high and admit nothing.

While the archive is below seed_n the novelty gate is OFF - bootstrap admits on
viability and liveness only. With it on from a cold start, reaching 256 entries
at a 15% target takes ~107 generations (five minutes at 2000 steps), and the
threshold would be adapting against no distribution at all.

Entries store the DECODED PHENOTYPE, never z. See
tests/test_physics_origin_roundtrip.py for why.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from services.novelty import (
    RejectsRing,
    knn_distances,
    knn_novelty,
    novelty_from_distances,
)


class AdaptiveThreshold:
    """Lehman-Stanley dynamic novelty threshold, driven to a target rate."""

    def __init__(self, initial: float = 0.05, target_rate: float = 0.15,
                 window: int = 100, lo: float = 1e-4, hi: float = 1.0,
                 step: float = 0.05):
        self.value = float(initial)
        self.target_rate = float(target_rate)
        self.window = int(window)
        self.lo = float(lo)
        self.hi = float(hi)
        self.step = float(step)
        self._recent: list[int] = []

    @property
    def rate(self) -> float:
        return (sum(self._recent) / len(self._recent)) if self._recent else 0.0

    def observe(self, admitted: bool) -> None:
        self._recent.append(1 if admitted else 0)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        if len(self._recent) < max(4, self.window // 10):
            return          # too little evidence to steer on
        if self.rate > self.target_rate:
            self.value *= (1.0 + self.step)
        elif self.rate < self.target_rate:
            self.value *= (1.0 - self.step)
        self.value = float(min(max(self.value, self.lo), self.hi))

    def state_dict(self) -> dict:
        return {"value": self.value, "recent": list(self._recent)}

    def load_state_dict(self, d: dict) -> None:
        self.value = float(d.get("value", self.value))
        self._recent = [int(x) for x in d.get("recent", [])]


@dataclass
class Candidate:
    """One tile's result for one generation, before the gates run."""
    brain: np.ndarray               # (10, 8) float32
    physics: np.ndarray             # (8,) float32, ABSOLUTE values
    embedding: np.ndarray           # (dim,) float32, unit norm
    liveness: float
    spec: str
    viable: bool = True
    gen: int = 0
    tile: int = 0
    run_id: str = ""
    goal: str = ""


@dataclass
class ArchiveEntry:
    id: int
    novelty: float
    liveness: float
    pinned: bool
    source: str
    spec: str
    goal: str
    run_id: str
    gen: int
    tile: int
    ts: float
    thumb: str = ""


class Archive:
    def __init__(self, store=None, capacity: int = 20000, k: int = 10,
                 liveness_min: float = 0.002, target_rate: float = 0.15,
                 seed_n: int = 256, dim: int = 512):
        self.store = store
        self.capacity = int(capacity)
        self.k = int(k)
        self.liveness_min = float(liveness_min)
        self.seed_n = int(seed_n)
        self._dim = int(dim)

        self.entries: list[ArchiveEntry] = []
        self.threshold = AdaptiveThreshold(target_rate=target_rate)
        self.rejects = RejectsRing(dim=self._dim)

        self._emb = np.zeros((0, self._dim), dtype=np.float32)
        self._brain = np.zeros((0, 10, 8), dtype=np.float32)
        self._phys = np.zeros((0, 8), dtype=np.float32)
        self._n = 0

        self._next_id = 0
        self._refresh_cursor = 0
        self._since_flush = 0
        self.n_nonfinite = 0
        self.n_rejected = 0
        self.blocked_by_pins = False

    # ---- views ---------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    @property
    def embeddings(self) -> np.ndarray:
        return self._emb[: self._n]

    @property
    def brains(self) -> np.ndarray:
        return self._brain[: self._n]

    @property
    def physics(self) -> np.ndarray:
        return self._phys[: self._n]

    def stats(self) -> dict:
        return {
            "size": self._n,
            "threshold": self.threshold.value,
            "admission_rate": self.threshold.rate,
            "n_nonfinite": self.n_nonfinite,
            "n_rejected": self.n_rejected,
            "n_pinned": sum(1 for e in self.entries if e.pinned),
            "blocked_by_pins": self.blocked_by_pins,
            "rejects_ring": len(self.rejects),
        }

    # ---- novelty -------------------------------------------------------

    def novelty_of(self, queries: np.ndarray) -> np.ndarray:
        """kNN novelty against archive UNION rejects ring.

        The two references are reduced separately and merged; concatenating a
        2k ring onto a 20k archive every generation would copy 40 MB - more
        than the novelty computation costs.
        """
        return novelty_from_distances(
            [knn_distances(queries, self.embeddings, self.k),
             knn_distances(queries, self.rejects.view(), self.k)],
            self.k,
        )

    def refresh(self, n: int) -> int:
        """Re-score n entries against the FULL archive, round-robin by position.

        Full archive, not a subsample: a subsampled neighbour set inflates kNN
        distances, so subsampled and full novelty values sit on different
        scales and could not be ranked against each other for parent sampling
        or eviction. This is also what makes eviction cheap - see _evict_one.
        """
        if n <= 0 or self._n == 0:
            return 0
        n = int(min(n, self._n))
        start = self._refresh_cursor % self._n
        idx = (np.arange(start, start + n) % self._n).astype(np.int64)
        nov = knn_novelty(self._emb[idx], self.embeddings, k=self.k,
                          exclude_self=True)
        for j, i in enumerate(idx):
            self.entries[int(i)].novelty = float(nov[j])
        self._refresh_cursor = int((start + n) % self._n)
        return n

    def centroid(self) -> np.ndarray | None:
        if self._n == 0:
            return None
        c = self.embeddings.mean(axis=0)
        nrm = float(np.linalg.norm(c))
        return (c / nrm).astype(np.float32) if nrm > 1e-8 else None

    def nearest(self, goal: np.ndarray) -> int | None:
        """Index of the entry most aligned with a goal embedding."""
        if self._n == 0:
            return None
        return int(np.argmax(self.embeddings @ np.asarray(goal, np.float32)))

    # ---- admission -----------------------------------------------------

    def consider(self, cand: Candidate, novelty: float, *, pinned: bool = False,
                 source: str = "expansion", thumb_crop=None) -> ArchiveEntry | None:
        """Run the gates and add on success. Returns the entry, or None."""
        if pinned:
            return self._add(cand, novelty, "pin", True, thumb_crop)

        if not np.isfinite(cand.embedding).all():
            self.n_nonfinite += 1
            self.n_rejected += 1
            return None

        if not cand.viable or cand.liveness < self.liveness_min:
            self._reject(cand)
            return None

        if self._n >= self.seed_n:
            ok = float(novelty) >= self.threshold.value
            self.threshold.observe(ok)
            if not ok:
                self._reject(cand)
                return None

        return self._add(cand, novelty, source, False, thumb_crop)

    def _reject(self, cand: Candidate) -> None:
        self.n_rejected += 1
        self.rejects.add(cand.embedding[None, :])

    def _add(self, cand, novelty, source, pinned, thumb_crop) -> ArchiveEntry | None:
        if self._n >= self.capacity and not self._evict_one():
            self.blocked_by_pins = True
            return None

        self._grow(1)
        i = self._n
        self._emb[i] = cand.embedding
        self._brain[i] = cand.brain
        self._phys[i] = cand.physics
        self._n += 1

        eid = self._next_id
        self._next_id += 1
        thumb = ""
        if thumb_crop is not None and self.store is not None:
            thumb = self.store.write_thumb(eid, thumb_crop)

        entry = ArchiveEntry(
            id=eid, novelty=float(novelty), liveness=float(cand.liveness),
            pinned=bool(pinned), source=source, spec=cand.spec, goal=cand.goal,
            run_id=cand.run_id, gen=int(cand.gen), tile=int(cand.tile),
            ts=time.time(), thumb=thumb,
        )
        self.entries.append(entry)
        if self.store is not None:
            self.store.append_index(asdict(entry))
        self._since_flush += 1
        return entry

    # ---- capacity ------------------------------------------------------

    def _evict_one(self) -> bool:
        """Drop the least novel non-pinned entry. -> did anything go?

        ASAL's illumination criterion (remove the least novel) applied as an
        eviction rule. It is O(N) and free BECAUSE refresh() keeps stored
        novelty current; a sampled nearest-neighbour pass here would be ~5
        GFLOP and would fire on every admission once the cap is reached.
        """
        best, best_v = None, np.inf
        for i, e in enumerate(self.entries):
            if not e.pinned and e.novelty < best_v:
                best, best_v = i, e.novelty
        if best is None:
            return False
        self._remove(best)
        return True

    def _remove(self, i: int) -> None:
        last = self._n - 1
        if i != last:
            self._emb[i] = self._emb[last]
            self._brain[i] = self._brain[last]
            self._phys[i] = self._phys[last]
            self.entries[i] = self.entries[last]
        self.entries.pop()
        self._n -= 1
        if self._refresh_cursor > self._n:
            self._refresh_cursor = 0

    def _grow(self, extra: int) -> None:
        need = self._n + int(extra)
        cap = len(self._emb)
        if need <= cap:
            return
        new_cap = max(256, need, cap * 2)

        def _re(a, tail):
            b = np.zeros((new_cap,) + tail, dtype=a.dtype)
            b[: self._n] = a[: self._n]
            return b

        self._emb = _re(self._emb, (self._dim,))
        self._brain = _re(self._brain, (10, 8))
        self._phys = _re(self._phys, (8,))

    # ---- persistence ---------------------------------------------------

    def maybe_flush(self, every: int = 200, force: bool = False) -> bool:
        if self.store is None:
            return False
        if not force and self._since_flush < int(every):
            return False
        ids = np.array([e.id for e in self.entries], dtype=np.int64)
        self.store.flush_vectors(ids, self.embeddings, self.brains, self.physics)
        self._since_flush = 0
        return True

    def load_from_store(self) -> tuple[int, int]:
        """-> (loaded, dropped). index.jsonl is the authority for WHICH entries
        exist; vectors.npz supplies their arrays. Keeping only the intersection
        means a crash between the last flush and quit costs the trailing
        entries, never the archive."""
        if self.store is None:
            return 0, 0
        rows, arrays = self.store.load()
        by_id = {int(r["id"]): r for r in rows if isinstance(r, dict) and "id" in r}
        if not arrays or not by_id:
            return 0, len(by_id)

        ids = np.asarray(arrays["ids"], dtype=np.int64)
        emb = np.asarray(arrays["embeddings"], dtype=np.float32)
        brains = np.asarray(arrays["brains"], dtype=np.float32)
        phys = np.asarray(arrays["physics"], dtype=np.float32)
        keep = [j for j, i in enumerate(ids) if int(i) in by_id]
        dropped = (len(ids) - len(keep)) + (len(by_id) - len(keep))

        self._n = 0
        self.entries = []
        self._grow(len(keep))
        for j in keep:
            i = self._n
            self._emb[i] = emb[j]
            self._brain[i] = brains[j]
            self._phys[i] = phys[j]
            self._n += 1
            r = by_id[int(ids[j])]
            self.entries.append(ArchiveEntry(
                id=int(r["id"]),
                novelty=float(r.get("novelty", 0.0)),
                liveness=float(r.get("liveness", 0.0)),
                pinned=bool(r.get("pinned", False)),
                source=str(r.get("source", "expansion")),
                spec=str(r.get("spec", "")),
                goal=str(r.get("goal", "")),
                run_id=str(r.get("run_id", "")),
                gen=int(r.get("gen", 0)),
                tile=int(r.get("tile", 0)),
                ts=float(r.get("ts", 0.0)),
                thumb=str(r.get("thumb", "")),
            ))
        self._next_id = max((e.id for e in self.entries), default=-1) + 1
        if dropped:
            print(f"[Archive] dropped {dropped} entries with no matching "
                  "index/vector row")
        return len(keep), dropped
