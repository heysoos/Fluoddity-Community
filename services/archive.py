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
    # The BRAIN this entry's genome belongs to, as a layout signature. Set at
    # load time from the directory it was read out of, and deliberately not
    # written to index.jsonl: the directory already says it, and adding a
    # column to an append-only file that older builds also write is a
    # migration for nothing.
    layout: str = ""
    # Which settings version admitted this entry. 0 for everything written
    # before the log existed, which is what a missing column reads as.
    cfg: int = 0


# Minimum cosine distance between two stored entries; 0 disables the rule.
# The unstructured-archive rule from quality-diversity (Cully & Mouret): store
# nothing within `l` of something already stored. Stops a converged expedition
# filling the archive with its own endpoint. See CLAUDE.md for the measured
# value and why this is not the same thing as the adaptive threshold in
# AdmissionRate.
DEFAULT_MIN_SEPARATION = 0.02


class Archive:
    def __init__(self, store=None, capacity: int = 20000, k: int = 10,
                 liveness_min: float = 0.002, dim: int | None = None,
                 min_separation: float = DEFAULT_MIN_SEPARATION,
                 layout=None, encoder: str | None = None):
        # seed_n lives on the driver (it picks bootstrap vs expansion), not
        # here - admission has no novelty gate to hold off.
        from services.brains import default_layout
        from services.vision_models import DEFAULT_KEY
        from services.vision_models import get as get_model

        # Which embedding space the stored vectors are in. An explicit dim
        # still wins: many call sites build a narrow Archive for speed.
        self.encoder = str(encoder or DEFAULT_KEY)
        self.encoder_mismatch = ""
        # The settings version in force. Read from the LOG, not from what this
        # session wrote, or reopening an archive rewrites version 0.
        self.cfg_version = max(0, store.latest_version()) if store else 0
        self._last_settings: dict | None = None
        if dim is None:
            dim = get_model(self.encoder).dim

        self.store = store
        self.capacity = int(capacity)
        self.k = int(k)
        self.liveness_min = float(liveness_min)
        self.min_separation = float(min_separation)
        self._dim = int(dim)
        # Falls back to the store's layout, then to Fourier, so the many call
        # sites that build a bare Archive() keep working unchanged.
        self.layout = (layout
                       or getattr(store, "layout", None)
                       or default_layout())

        self.entries: list[ArchiveEntry] = []
        self.admission = AdmissionRate()
        self.rejects = RejectsRing(dim=self._dim)

        self._emb = np.zeros((0, self._dim), dtype=np.float32)
        # FLAT and layout-wide. (10, 8) was Fourier's own structure, which no
        # other modality has - Gabor is 14 floats a filter, MLP is not a grid at
        # all. Every consumer already flattens before use (genome_spec.encode
        # reshapes internally), so the 2D form bought nothing.
        #
        # ONE array, as wide as the widest layout present, rather than an array
        # per layout with a row map: a brain is at most MAX_BRAIN_FLOATS wide,
        # so the padding a mixed archive carries is bounded and small, and the
        # swap-with-last in _remove stays a single assignment. The width grows
        # when a wider layout arrives; an archive of one layout is exactly as
        # wide as it has always been. brain_at() trims to the entry's own
        # length, so nothing downstream ever sees the padding.
        self._bw = int(self.layout.length)
        self._brain = np.zeros((0, self._bw), dtype=np.float32)
        self._phys = np.zeros((0, 8), dtype=np.float32)
        self._n = 0

        # Per SIGNATURE, because an id is unique within the directory that
        # holds it and nowhere else. Two layouts each hold an entry 0.
        self._next_ids: dict[str, int] = {}
        # Signature -> the store that owns those entries' index, vectors and
        # thumbnails. self.store is the running layout's, and is in here too.
        self._stores: dict[str, object] = {}
        # Signature -> genome width, learned from the layout or from the array
        # that was loaded. Parsing it back out of a signature string would be
        # guessing at a format that belongs to the modality.
        self._widths: dict[str, int] = {}
        if store is not None:
            sig = self.signature
            self._stores[sig] = store
            self._widths[sig] = int(self.layout.length)
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
    def signature(self) -> str:
        """The RUNNING layout. Entries of any other are read-only here."""
        return self.layout.signature()

    def layout_at(self, i: int) -> str:
        """The signature of entry `i`'s brain."""
        return self.entries[i].layout or self.signature

    def is_native(self, i: int) -> bool:
        """Can the running brain decode entry `i`? Everything that turns a
        stored genome back into a creature has to ask."""
        return self.layout_at(i) == self.signature

    def native_rows(self) -> np.ndarray:
        """Row indices the running brain can decode, in archive order.

        Parent sampling and seed selection filter through this: novelty ranks
        across every brain, because it is about pictures, but a parent has to
        be decodable by the optimizer that will mutate it.
        """
        sig = self.signature
        return np.array([i for i, e in enumerate(self.entries)
                         if (e.layout or sig) == sig], dtype=np.int64)

    @property
    def stores(self) -> dict:
        """Signature -> the store owning those entries. The thumbnail loader
        resolves through this, because a filename alone names one per brain."""
        return self._stores

    def thumb_key(self, i: int) -> str:
        """ThumbCache key for entry `i`. Every brain has a 000000.jpg."""
        t = self.entries[i].thumb
        return f"{self.layout_at(i)}/{t}" if t else ""

    def brain_at(self, i: int) -> np.ndarray:
        """Entry `i`'s genome, trimmed to its own layout's width."""
        w = self._widths.get(self.layout_at(i), self._bw)
        return self._brain[i, :w]

    @property
    def brains(self) -> np.ndarray:
        """Every genome, padded to the widest layout present.

        Only persistence wants this. Anything that decodes must use brain_at(),
        which trims to the entry's own width - a padded Gabor vector run
        through Fourier's squash is a different creature, silently.
        """
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
        # Admission always writes the RUNNING brain: a candidate came off this
        # generation's tiles.
        sig = self.signature
        self._grow(1)
        i = self._n
        self._emb[i] = cand.embedding
        # Flattened on the way in: Fourier candidates arrive as (10, 8) because
        # genome_spec.decode preserves that shape for its legacy callers.
        flat = np.asarray(cand.brain, dtype=np.float32).reshape(-1)
        self._brain[i] = 0.0
        self._brain[i, :len(flat)] = flat
        self._phys[i] = cand.physics
        self._n += 1

        eid = self._next_ids.get(sig, 0)
        self._next_ids[sig] = eid + 1
        thumb = ""
        if thumb_crop is not None and self.store is not None:
            thumb = self.store.write_thumb(eid, thumb_crop)

        entry = ArchiveEntry(
            id=eid, novelty=float(novelty), liveness=float(cand.liveness),
            pinned=bool(pinned), source=source, spec=cand.spec, goal=cand.goal,
            run_id=cand.run_id, gen=int(cand.gen), tile=int(cand.tile),
            ts=time.time(), thumb=thumb, layout=sig, cfg=int(self.cfg_version),
        )
        self.entries.append(entry)
        if self.store is not None:
            # WITHOUT `layout`: the directory the row lands in already says it,
            # and older builds append to the same file.
            row = asdict(entry)
            row.pop("layout", None)
            self.store.append_index(row)
        self._since_flush += 1
        self.revision += 1
        return entry

    def record_settings(self, current: dict, gen: int) -> int:
        """Append a settings version if anything moved. -> the version in force.

        Called once per generation and once more when the archive closes, never
        per frame. The encoder rides along so the log is self-contained.
        """
        from services.settings_history import diff, replay

        if self.store is None:
            return self.cfg_version
        data = dict(current)
        data["encoder"] = self.encoder

        if self._last_settings is None:
            rows = self.store.load_history()
            if not rows:
                self.store.append_history({
                    "v": 0, "ts": time.time(), "gen": int(gen),
                    "entries": len(self.entries), "full": data})
                self._last_settings = data
                self.cfg_version = 0
                return 0
            self._last_settings = replay(rows)

        moved = diff(self._last_settings, data)
        if not moved:
            return self.cfg_version
        self.cfg_version = self.store.latest_version() + 1
        self.store.append_history({
            "v": self.cfg_version, "ts": time.time(), "gen": int(gen),
            "entries": len(self.entries), "changed": moved})
        self._last_settings = data
        return self.cfg_version

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
        # From the entry's OWN store: pruning ranks across brains, so the row
        # being evicted need not belong to the one that is running, and
        # deleting through self.store would unlink some other brain's picture
        # that happens to share the filename.
        owner = self._stores.get(self.layout_at(i))
        if owner is not None:
            owner.delete_thumb(self.entries[i].thumb)
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

    def _widen(self, width: int) -> None:
        """Make room for a layout wider than any seen so far.

        Padding, never reinterpretation: existing rows keep their values at
        their own width and brain_at() trims each back to it.
        """
        width = int(width)
        if width <= self._bw:
            return
        b = np.zeros((len(self._brain), width), dtype=self._brain.dtype)
        b[:, : self._bw] = self._brain
        self._brain = b
        self._bw = width

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
        self._brain = _re(self._brain, (self._bw,))
        self._phys = _re(self._phys, (8,))

    # ---- persistence ---------------------------------------------------

    def maybe_flush(self, every: int = 200, force: bool = False) -> bool:
        """Rewrite the vector arrays, ONE FILE PER LAYOUT.

        A signature directory holds only its own brain's entries, at its own
        genome width, so a single-layout archive's vectors.npz is byte-for-byte
        what it has always been. Every layout is rewritten, not just the
        running one: pruning evicts across brains, so a generation can remove
        rows from a layout nothing has admitted to.
        """
        if self.store is None:
            return False
        if not force and self._since_flush < int(every):
            return False
        rows: dict[str, list[int]] = {}
        for i in range(self._n):
            rows.setdefault(self.layout_at(i), []).append(i)
        for sig, store in self._stores.items():
            idx = np.array(rows.get(sig, []), dtype=np.int64)
            w = self._widths.get(sig, self._bw)
            store.flush_vectors(
                np.array([self.entries[i].id for i in idx], dtype=np.int64),
                self._emb[idx],
                self._brain[idx][:, :w],
                self._phys[idx],
                np.array([self.entries[i].novelty for i in idx],
                         dtype=np.float32),
            )
        self._since_flush = 0
        return True

    def load_from_store(self) -> tuple[int, int]:
        """Load EVERY brain's entries under this archive. -> (loaded, dropped).

        An archive is a library of pictures with a genome attached, and only
        the genome is per-brain: novelty, separation, admission and the map all
        run on the embedding. So one archive holds every layout, and only the
        four things that decode ask which.

        The running layout is loaded first, so a single-layout archive keeps
        exactly the row order it has always had.
        """
        self.encoder_mismatch = ""
        if self.store is None:
            return 0, 0
        # Nothing downstream compares encoders: at equal width a foreign vector
        # is silently wrong rather than an error, so this is the only place the
        # mistake is catchable.
        if self.store.encoder != self.encoder:
            self.encoder_mismatch = (
                f"this archive was built with {self.store.encoder}; "
                f"the search is running {self.encoder}")
            print(f"[Archive] {self.encoder_mismatch}; nothing loaded")
            return 0, 0
        from services.archive_io import ArchiveStore, signature_dirs

        self._n = 0
        self.entries = []
        loaded = dropped = 0
        sigs = [self.signature] + [
            p.name for p in signature_dirs(self.store.base)
            if p.name != self.signature]
        for sig in sigs:
            store = self._stores.get(sig)
            if store is None:
                store = ArchiveStore(self.store.base, signature=sig)
                self._stores[sig] = store
            got, lost = self._load_one(sig, store)
            loaded += got
            dropped += lost

        # Unconditional, and now ACROSS brains: an entry's stored novelty was
        # scored against however much archive existed at admission, under one
        # layout. See CLAUDE.md.
        self.rescore_all()
        if dropped:
            print(f"[Archive] dropped {dropped} entries with no matching "
                  "index/vector row")
        return loaded, dropped

    def _load_one(self, sig: str, store) -> tuple[int, int]:
        """Append one layout directory's entries. -> (loaded, dropped)."""
        rows, arrays = store.load()
        by_id = {int(r["id"]): r for r in rows if isinstance(r, dict) and "id" in r}
        # From the INDEX (every id ever issued under this layout), not from the
        # surviving entries - else the counter restarts at the first lost id
        # and the next run re-issues ids that already exist. See CLAUDE.md.
        self._next_ids[sig] = max(by_id, default=-1) + 1
        if not arrays or not by_id:
            return 0, len(by_id)

        ids = np.asarray(arrays["ids"], dtype=np.int64)
        emb = np.asarray(arrays["embeddings"], dtype=np.float32)
        # Archives written before brain modalities stored (N, 10, 8); flatten so
        # both forms load. A width that does not match the layout is a genuine
        # mismatch and the entries are dropped rather than reinterpreted -
        # though the signature in the path should make that unreachable.
        brains = np.asarray(arrays["brains"], dtype=np.float32)
        brains = brains.reshape(len(brains), -1) if len(brains) else brains
        want = self._widths.get(sig)
        if want is not None and brains.shape[1:] != (want,):
            print(f"[Archive] {sig} brains are {brains.shape[1:]}, that layout "
                  f"wants ({want},); ignoring the stored brains")
            brains = np.zeros((len(ids), want), dtype=np.float32)
        # A sibling layout's width is whatever it stored: nothing here can
        # rebuild its BrainLayout from a directory name, and nothing needs to -
        # brain_at() trims to this and only the modality decodes it.
        width = int(brains.shape[1]) if brains.size else int(want or 0)
        self._widths.setdefault(sig, width)
        self._widen(width)
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

        self._grow(len(keep))
        for j in keep:
            i = self._n
            self._emb[i] = emb[j]
            self._brain[i] = 0.0
            self._brain[i, :brains.shape[1]] = brains[j]
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
                layout=sig,
                cfg=int(r.get("cfg", 0)),
            ))
        return len(keep), dropped
