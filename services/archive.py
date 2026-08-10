"""The exploration archive: admission gates, capacity, and novelty bookkeeping.

ADMIT GENEROUSLY, PRUNE AFTERWARDS. Two gates, and both ask only whether the
tile is a picture of something:

  viable   - the tile is not black or blown out
  alive    - liveness >= liveness_min (ASAL Eq.3; a frozen canvas scores ~0)

Novelty is not a gate. It ranks, and capacity evicts the bottom of the ranking
(prune_to_capacity). The asymmetry is the argument: a rejected pattern is gone
for good and cost a full 2000-step rollout to produce, while an admitted dud
costs one slot until something more novel displaces it.

There WAS a third gate, an adaptive kNN-novelty threshold driven to a target
admission rate. It was removed 2026-08-08 for two independent reasons, both
measured - the controller could not be stabilised at any gain, and a
generation's tiles are not independent draws so they clear or miss any bar
together. See AdmissionRate.

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
    nearest_distance,
    novelty_from_distances,
)


class AdmissionRate:
    """What fraction of recent candidates got in. A READOUT, not a controller.

    It replaces AdaptiveThreshold, a Lehman-Stanley novelty gate driven to a
    target rate, which was removed 2026-08-08 because it could not be made
    stable on this substrate. Its `observe()` ran once per CANDIDATE - 16 tiles
    a generation at grid 4, 64 at grid 8 - and each observation multiplied the
    threshold by 1.05 or 0.95. All 16 pushing the same way moves it 2.18x in a
    single generation (22.7x at grid 8), while the rate it steers on is
    averaged over the last 100 observations, i.e. roughly six generations old.
    Gain that far above the measurement lag is a limit cycle, not a controller:
    simulated on a STATIONARY novelty distribution with no archive at all, it
    admitted nothing in 61% of generations and ran an admission-rate standard
    deviation of 0.315 against a Bernoulli noise floor of 0.089. That matched
    the real archives, where admission was bimodal rather than near target.

    Novelty now prunes instead of gating: everything viable and alive is
    admitted, and capacity evicts the least novel. Rate is kept only so the
    "nothing is getting in" warning still has something to look at.
    """

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


# Minimum cosine distance between two stored entries. 0 disables the rule.
#
# The unstructured-archive rule from quality-diversity (Cully & Mouret): store
# nothing within `l` of something already stored, so the archive is a covering
# of the space rather than a log of everything that happened. It is what stops a
# converged expedition filling the archive with its own endpoint - measured on
# the real archives, ONE goal had already contributed 25% of debug07 (3200 of
# 12672) and 30% of debug05.
#
# 0.02 is measured, not chosen. Replaying each archive in insertion order:
#
#   archive    1-NN median   kept at l=0.02   of that flood   of the rest
#   default        0.0162         2604/4808         33.3%         62.6%
#   debug05        0.0113         2717/8002         20.6%         39.7%
#   debug07        0.0074        1402/12672          8.3%         12.0%
#
# so it is selective - it always cuts the flood about twice as hard as the rest
# - and it takes a generation's admissions from 16-64 down to 5-14. Larger
# values are not a matter of taste: at 0.05 every archive keeps under 6%.
#
# Note what this is NOT. The adaptive novelty threshold removed 2026-08-08
# failed because it was a feedback controller whose gain ran far ahead of its
# measurement lag, and because a generation's tiles are correlated so they clear
# or miss any bar together. Neither applies here: there is no controller and no
# gain, and correlated tiles landing on top of each other is exactly the case
# this is meant to reject. See AdmissionRate.
DEFAULT_MIN_SEPARATION = 0.02


class Archive:
    def __init__(self, store=None, capacity: int = 20000, k: int = 10,
                 liveness_min: float = 0.002, dim: int = 512,
                 min_separation: float = DEFAULT_MIN_SEPARATION):
        # seed_n is gone from here: it existed only to hold the novelty gate
        # off during bootstrap, and there is no novelty gate. The DRIVER still
        # has one - it chooses bootstrap vs expansion - but that is a question
        # about the search, not about admission.
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
        # Bumped by anything that changes what a viewer would draw: which
        # entries exist, or their novelty. The browser derives a PCA projection
        # and a sort order from those, both O(n) and both unchanged between
        # generations - without a version to compare, the only way to know they
        # are still valid is to recompute them every frame, which is what made
        # the browser cost ~10x the frame time. Monotonic, never reset.
        self.revision = 0
        self.n_nonfinite = 0
        self.n_rejected = 0
        # Split by reason, because they mean opposite things. "Dead" rising is a
        # fault - a black capture, a frozen preset, a liveness floor set too
        # high. "Too close" rising is the separation rule working, and during a
        # converged expedition it should be nearly every tile.
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
        """Mean stored kNN novelty: how far apart the archive's entries are.

        With `size`, this is the pair that says whether exploration is still
        finding new territory - a growing archive whose mean novelty is flat is
        spreading, one whose mean novelty falls is filling in. Free, because
        refresh() keeps the column current for its own reasons.
        """
        if not self.entries:
            return 0.0
        return float(np.mean([e.novelty for e in self.entries]))

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
        or eviction. This is also what makes eviction cheap - see
        prune_to_capacity.
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
        but means a stored novelty is only ever as fresh as the last sweep -
        and nothing at all after a reload, because the value on disk was
        measured at admission time against however much archive existed then.
        Entry #50 was scored against 49 neighbours and entry #4000 against
        3999; those numbers are not on the same scale and the search compares
        them as if they were. Measured 2026-08-08 on the default archive, the
        stored column correlates 0.075 with a correct rescore.

        The whole sweep is one blocked matmul - 0.23 s at 4808 entries, 4.3 s
        at the 20000 capacity - so it is affordable on load, which is the one
        moment the round-robin cannot cover.
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

        Three gates, in order of what they mean:

          finite/viable/alive   is this a picture of something at all
          separation            do we already have one of these

        There is still NO novelty gate, and the difference matters. A novelty
        threshold asks "is this neighbourhood empty enough", which is a moving,
        archive-wide judgement - that is what was removed 2026-08-08 and it is
        not coming back (see AdmissionRate). Separation asks the local,
        parameter-free question "is there already an entry within l of this
        one", which is the unstructured-archive rule from quality-diversity.

        `force` bypasses separation only. The caller uses it to keep the best
        tile of every generation whatever happens, so a converged expedition
        still leaves a trail rather than vanishing from the record entirely.

        `ignore_liveness` drops the CHANGE half of the alive gate, and nothing
        else - `cand.viable` (is this a black or blown-out frame) still has to
        hold. Two callers pass it: the expedition summit and a goal record.
        Liveness is a floor on the bulk of the archive, not a veto over a
        chosen entry, and it is measurably the wrong test for these two -
        liveness is HIGHER during the transient after a reset than once a
        pattern settles into its attractor, which is precisely what a
        converging expedition produces.

        `separation` is passed in because the caller has the whole batch and can
        do one matmul for all of it; omitted, it is computed here.
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
                # Deliberately NOT added to the rejects ring. The ring exists so
                # novelty remembers regions the search was refused; a
                # separation rejection means the region is already IN the
                # archive, which novelty measures against anyway. Feeding these
                # in would evict the ring's real content - the dead regions -
                # within a couple of generations at grid 8.
                self.admission.observe(False)
                return None

        entry = self._add(cand, novelty, source, False, thumb_crop)
        self.admission.observe(entry is not None)
        return entry

    def _reject(self, cand: Candidate) -> None:
        self.n_rejected += 1
        self.rejects.add(cand.embedding[None, :])

    def _add(self, cand, novelty, source, pinned, thumb_crop) -> ArchiveEntry | None:
        # Deliberately no eviction here. Pruning is a once-per-generation bulk
        # pass (prune_to_capacity), so len() may exceed capacity by up to one
        # generation - 64 entries at grid 8 - and never by more.
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

        The ONLY pruning rule now that admission does not gate on novelty:
        ASAL's illumination criterion as an eviction rule, so what survives is
        the `capacity` most novel things the search has found. This is where
        the compute the old threshold used to throw away gets spent instead -
        every pattern is scored and kept, and only then ranked.

        Bulk rather than one-at-a-time-on-admission: a generation adds 16-64
        entries at once and one argpartition costs what one linear scan did.
        It runs AFTER refresh() so it ranks on the freshest novelty available;
        that is also why the persisted novelty column has to be live, since
        eviction is only meaningful if the numbers it compares are.
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
        # Descending, because _remove swaps the LAST entry into the hole. Going
        # highest-first means every entry moved down is one we are keeping, so
        # no victim is ever relocated out from under the loop.
        for i in sorted((int(v) for v in victims), reverse=True):
            self._remove(i)
        self.n_evicted += take
        return take

    def _remove(self, i: int) -> None:
        # The picture goes with the entry. Nothing could reach it afterwards
        # anyway: index.jsonl is append-only and the id is gone from
        # vectors.npz, so load_from_store drops the row on the next open.
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
        # vectors.npz is the authority for novelty when it carries it: the
        # index row can only ever hold the at-admission value. Archives written
        # before 2026-08-08 have no such array and fall back to the row.
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
        self._next_id = max((e.id for e in self.entries), default=-1) + 1
        # Unconditional, even when the file carried a novelty array: the array
        # is only as fresh as the last round-robin sweep, and load is the one
        # moment a whole-archive pass is both affordable and necessary. This is
        # what stops the search reopening on stale scores - notably generation
        # 0, whose entries are all stamped 1.0 by the no-reference convention
        # and would otherwise take 100% of the p ~ novelty^4 parent weight.
        self.rescore_all()
        if dropped:
            print(f"[Archive] dropped {dropped} entries with no matching "
                  "index/vector row")
        return len(keep), dropped
