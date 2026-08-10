"""The exploration archive: admission gates, capacity, and novelty bookkeeping.

Admission gates only on `viable` (not black/blown out) and `alive` (liveness
>= liveness_min). Novelty is not a gate - it ranks, and capacity evicts the
bottom of the ranking (prune_to_capacity). See CLAUDE.md and AdmissionRate for
why there is no adaptive novelty threshold.

Entries store the DECODED PHENOTYPE, never z. See
tests/test_physics_origin_roundtrip.py.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from services.novelty import (
    RejectsRing,
    knn_distances,
    knn_novelty,
    nearest_distance,
    novelty_from_distances,
)


class AdmissionRate:
    """What fraction of recent candidates got in. A READOUT, not a controller
    - there is no adaptive novelty gate. See CLAUDE.md."""

    def __init__(self, window: int = 100):
        self.window = int(window)
        self._recent: list[int] = []

    @property
    def rate(self) -> float:
        return (sum(self._recent) / len(self._recent)) if self._recent else 0.0

    def observe(self, admitted: bool) -> None:
        self._recent.append(1 if admitted else 0)
        if len(self._recent) > self.window:
            self._recent.pop(0)


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


# Minimum cosine distance between two stored entries; 0 disables the rule.
# The unstructured-archive rule from quality-diversity (Cully & Mouret): store
# nothing within `l` of something already stored. Stops a converged expedition
# filling the archive with its own endpoint. See CLAUDE.md for the measured
# value and why this is not the same thing as the adaptive threshold in
# AdmissionRate.
DEFAULT_MIN_SEPARATION = 0.02


class Archive:
    def __init__(self, store=None, capacity: int = 20000, k: int = 10,
                 liveness_min: float = 0.002, dim: int = 512,
                 min_separation: float = DEFAULT_MIN_SEPARATION):
        # seed_n lives on the driver (it picks bootstrap vs expansion), not
        # here - admission has no novelty gate to hold off.
        self.store = store
        self.capacity = int(capacity)
        self.k = int(k)
        self.liveness_min = float(liveness_min)
        self.min_separation = float(min_separation)
        self._dim = int(dim)

        self.entries: list[ArchiveEntry] = []
        self.admission = AdmissionRate()
        self.rejects = RejectsRing(dim=self._dim)

        self._emb = np.zeros((0, self._dim), dtype=np.float32)
        self._brain = np.zeros((0, 10, 8), dtype=np.float32)
        self._phys = np.zeros((0, 8), dtype=np.float32)
        self._n = 0

        self._next_id = 0
        self._refresh_cursor = 0
        self._since_flush = 0
        # Bumped by anything that changes what a viewer would draw (entries or
        # their novelty), so a viewer can cache derived views instead of
        # recomputing every frame. Monotonic, never reset.
        self.revision = 0
        self.n_nonfinite = 0
        self.n_rejected = 0
        # Split by reason: "dead" rising is a fault, "too close" rising is the
        # separation rule working as intended.
        self.n_rejected_dead = 0
        self.n_rejected_close = 0
        self.n_evicted = 0
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
            "capacity": self.capacity,
            "admission_rate": self.admission.rate,
            "n_nonfinite": self.n_nonfinite,
            "n_rejected": self.n_rejected,
            "n_rejected_dead": self.n_rejected_dead,
            "n_rejected_close": self.n_rejected_close,
            "n_evicted": self.n_evicted,
            "n_pinned": sum(1 for e in self.entries if e.pinned),
            "blocked_by_pins": self.blocked_by_pins,
            "rejects_ring": len(self.rejects),
            "min_separation": self.min_separation,
            "mean_novelty": self.mean_novelty(),
        }

    def mean_novelty(self) -> float:
        """Mean stored kNN novelty: how far apart the archive's entries are."""
        if not self.entries:
            return 0.0
        return float(np.mean([e.novelty for e in self.entries]))

    # ---- novelty -------------------------------------------------------

    def novelty_of(self, queries: np.ndarray) -> np.ndarray:
        """kNN novelty against archive UNION rejects ring.

        The two references are reduced separately and merged rather than
        concatenated, to avoid copying the whole archive every generation.
        """
        return novelty_from_distances(
            [knn_distances(queries, self.embeddings, self.k),
             knn_distances(queries, self.rejects.view(), self.k)],
            self.k,
        )

    def refresh(self, n: int) -> int:
        """Re-score n entries against the FULL archive, round-robin by position.

        Full archive, not a subsample: subsampled kNN distances sit on a
        different scale and could not be ranked against full ones for parent
        sampling or eviction.
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
        self.revision += 1
        return n

    def rescore_all(self) -> int:
        """Re-score EVERY entry against the whole archive. -> how many.

        refresh() spreads this over generations, which is right during a run
        but leaves the on-disk novelty column stale after a reload - each
        entry was scored against however much archive existed at admission
        time, so the values are not on a comparable scale. See CLAUDE.md.
        """
        if self._n == 0:
            return 0
        nov = knn_novelty(self.embeddings, self.embeddings, k=self.k,
                          exclude_self=True)
        for i, e in enumerate(self.entries):
            e.novelty = float(nov[i])
        self._refresh_cursor = 0
        self.revision += 1
        return self._n

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

    def separation_of(self, queries: np.ndarray) -> np.ndarray:
        """Cosine distance from each query to the closest STORED entry.

        Archive only, not archive union rejects: the question is "do we already
        have one of these", and a rejected pattern is precisely one we do not.
        """
        return nearest_distance(queries, self.embeddings)

    # ---- admission -----------------------------------------------------

    def consider(self, cand: Candidate, novelty: float, *, pinned: bool = False,
                 source: str = "expansion", thumb_crop=None,
                 separation: float | None = None,
                 force: bool = False,
                 ignore_liveness: bool = False) -> ArchiveEntry | None:
        """Run the gates and add on success. Returns the entry, or None.

        Gates, in order: finite/viable/alive (is this a picture of something
        at all), then separation (is there already an entry within
        `min_separation` of this one). No novelty gate - see AdmissionRate.

        `force` bypasses separation only, so the caller can keep one tile per
        generation regardless.

        `ignore_liveness` drops the CHANGE half of the alive gate;
        `cand.viable` (black/blown-out) still applies. Used by the expedition
        summit and goal records, where liveness is the wrong test - see
        CLAUDE.md.

        `separation` lets the caller batch the matmul over the whole
        generation; omitted, it is computed here.
        """
        if pinned:
            return self._add(cand, novelty, "pin", True, thumb_crop)

        if not np.isfinite(cand.embedding).all():
            self.n_nonfinite += 1
            self.n_rejected += 1
            self.n_rejected_dead += 1
            self.admission.observe(False)
            return None

        alive = ignore_liveness or cand.liveness >= self.liveness_min
        if not cand.viable or not alive:
            self.n_rejected_dead += 1
            self._reject(cand)
            self.admission.observe(False)
            return None

        if self.min_separation > 0.0 and not force:
            d = (float(separation) if separation is not None
                 else float(self.separation_of(cand.embedding[None, :])[0]))
            if d < self.min_separation:
                self.n_rejected += 1
                self.n_rejected_close += 1
                # Not added to the rejects ring: that region is already IN the
                # archive, which novelty measures against anyway.
                self.admission.observe(False)
                return None

        entry = self._add(cand, novelty, source, False, thumb_crop)
        self.admission.observe(entry is not None)
        return entry

    def _reject(self, cand: Candidate) -> None:
        self.n_rejected += 1
        self.rejects.add(cand.embedding[None, :])

    def _add(self, cand, novelty, source, pinned, thumb_crop) -> ArchiveEntry | None:
        # No eviction here - pruning is a once-per-generation bulk pass
        # (prune_to_capacity), so len() may exceed capacity briefly.
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
        self.revision += 1
        return entry

    # ---- capacity ------------------------------------------------------

    def prune_to_capacity(self) -> int:
        """Drop the least novel entries until len <= capacity. -> how many went.

        The only pruning rule: what survives is the `capacity` most novel
        things found. Bulk rather than per-admission, and run AFTER refresh()
        so it ranks on the freshest novelty available.
        """
        over = self._n - int(self.capacity)
        if over <= 0:
            self.blocked_by_pins = False
            return 0

        nov = np.array([e.novelty for e in self.entries], dtype=np.float64)
        pinned = np.array([e.pinned for e in self.entries], dtype=bool)
        nov[pinned] = np.inf
        take = int(min(over, int((~pinned).sum())))
        self.blocked_by_pins = take < over
        if take <= 0:
            return 0

        victims = np.argpartition(nov, take - 1)[:take]
        # Descending: _remove swaps the last entry into the hole, so removing
        # highest-index-first never relocates an unprocessed victim.
        for i in sorted((int(v) for v in victims), reverse=True):
            self._remove(i)
        self.n_evicted += take
        return take

    def _remove(self, i: int) -> None:
        # Thumbnail goes with the entry - nothing could reach it afterwards.
        if self.store is not None:
            self.store.delete_thumb(self.entries[i].thumb)
        last = self._n - 1
        if i != last:
            self._emb[i] = self._emb[last]
            self._brain[i] = self._brain[last]
            self._phys[i] = self._phys[last]
            self.entries[i] = self.entries[last]
        self.entries.pop()
        self._n -= 1
        self.revision += 1
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
        nov = np.array([e.novelty for e in self.entries], dtype=np.float32)
        self.store.flush_vectors(ids, self.embeddings, self.brains,
                                 self.physics, nov)
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
        # vectors.npz is the authority for novelty when it carries it - the
        # index row only ever holds the at-admission value. Older archives
        # lack the array and fall back to the row.
        nov = arrays.get("novelty")
        nov = (np.asarray(nov, dtype=np.float32)
               if nov is not None and len(np.asarray(nov)) == len(ids)
               else None)
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
                novelty=(float(nov[j]) if nov is not None
                         else float(r.get("novelty", 0.0))),
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
        # From the INDEX (every id ever issued), not from the surviving
        # entries (only those still backed by vectors.npz) - else the counter
        # restarts at the first lost id and the next run re-issues ids that
        # already exist, overwriting those entries' thumbnails. See CLAUDE.md.
        self._next_id = max(by_id, default=-1) + 1
        # Unconditional even when the file carried a novelty array: only a
        # whole-archive pass is guaranteed fresh, and load is the one moment
        # it is affordable. See CLAUDE.md.
        self.rescore_all()
        if dropped:
            print(f"[Archive] dropped {dropped} entries with no matching "
                  "index/vector row")
        return len(keep), dropped
