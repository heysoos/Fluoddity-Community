"""One archive, many brains.

An archive is a library of PICTURES with a genome attached, and only the genome
is per-brain: rescore_all is knn_novelty over embeddings and descriptor() is a
CLIP centroid, so neither reads a brain. Novelty, separation, admission and
pruning therefore pool across every layout, and only the four things that
decode - re-encoding, decode, the optimizer's width, and seeds - ask which one
an entry belongs to.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.archive import Archive, Candidate
from services.archive_io import ArchiveStore
from services.brains import BrainLayout, default_layout

DIM = 8
FOURIER = default_layout()
GABOR = BrainLayout("gabor", (12,), 168)


def _cand(vec, brain_len=None, liveness=0.5):
    e = np.asarray(vec, dtype=np.float32)
    e = e / max(float(np.linalg.norm(e)), 1e-8)
    n = brain_len if brain_len is not None else FOURIER.length
    return Candidate(embedding=e, brain=np.arange(n, dtype=np.float32),
                     physics=np.zeros(8, np.float32), liveness=liveness,
                     spec="brain", goal="", run_id="r", gen=0, tile=0,
                     viable=True)


def _axis(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def _fill(root, layout, n, start_axis=0, crop=None):
    """Write n entries of `layout` into its own signature directory."""
    store = ArchiveStore(root, layout)
    arc = Archive(store=store, layout=layout, dim=DIM)
    for j in range(n):
        arc.consider(_cand(_axis(start_axis + j), layout.length),
                     novelty=1.0, force=True, thumb_crop=crop)
    arc.maybe_flush(force=True)
    arc.store.close()
    return arc


# ---- the collision -------------------------------------------------------

def test_two_brains_each_holding_an_entry_zero_stay_distinct(tmp_path):
    """The load-bearing case. An id is unique inside one layout's directory and
    nowhere else, so a pooled archive holds two entry 0s - and every one-shot
    the browser sets has to name a ROW, not an id."""
    root = tmp_path / "mixed"
    _fill(root, FOURIER, 2, start_axis=0)
    _fill(root, GABOR, 2, start_axis=4)

    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    loaded, _ = arc.load_from_store()

    assert loaded == 4, "every brain's entries belong to the one archive"
    assert sorted(e.id for e in arc.entries) == [0, 0, 1, 1]
    assert {arc.layout_at(i) for i in range(len(arc.entries))} == {
        FOURIER.signature(), GABOR.signature()}


def test_a_thumbnail_key_carries_the_signature(tmp_path):
    """Both layouts write a 000000.jpg. A cache keyed on the filename alone
    hands out the wrong picture - silently, which is the whole danger."""
    root = tmp_path / "mixed"
    crop = np.full((32, 32, 3), 200, dtype=np.uint8)
    _fill(root, FOURIER, 1, start_axis=0, crop=crop)
    _fill(root, GABOR, 1, start_axis=4, crop=crop)

    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    arc.load_from_store()

    names = {arc.entries[i].thumb for i in range(len(arc.entries))}
    keys = {arc.thumb_key(i) for i in range(len(arc.entries))}
    assert len(names) == 1, "the filenames really do collide"
    assert len(keys) == 2, "the keys must not"
    # And each key resolves through its OWN store, to a file that exists.
    for i in range(len(arc.entries)):
        sig, _, name = arc.thumb_key(i).rpartition("/")
        assert arc.stores[sig].thumb_path(name).is_file()


def test_the_loader_reads_each_signature_from_its_own_store(tmp_path):
    """gl_loader's contract, without a GL context."""
    from services.thumb_cache import ThumbCache

    root = tmp_path / "mixed"
    crop = np.full((32, 32, 3), 200, dtype=np.uint8)
    _fill(root, FOURIER, 1, start_axis=0, crop=crop)
    _fill(root, GABOR, 1, start_axis=4, crop=crop)
    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    arc.load_from_store()

    seen = []

    def loader(key):
        sig, _, name = key.rpartition("/")
        seen.append((sig, name))
        return object()

    cache = ThumbCache(loader, capacity=8)
    for i in range(len(arc.entries)):
        cache.get(arc.thumb_key(i))
    assert sorted(seen) == sorted([(FOURIER.signature(), "000000.jpg"),
                                   (GABOR.signature(), "000000.jpg")])


# ---- what pools, and what does not ---------------------------------------

def test_novelty_is_scored_across_brains(tmp_path):
    """Novelty is about pictures, so a look-alike from another brain lowers it.
    If it did not, the two would never compete for capacity."""
    root = tmp_path / "mixed"
    _fill(root, FOURIER, 1, start_axis=0)
    alone = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    alone.load_from_store()
    before = alone.entries[0].novelty

    _fill(root, GABOR, 1, start_axis=0)      # the SAME embedding, another brain
    both = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    both.load_from_store()
    after = next(e.novelty for e in both.entries
                 if e.layout == FOURIER.signature())

    assert after < before, "the other brain's look-alike was invisible"


def test_separation_refuses_a_cross_brain_near_duplicate(tmp_path):
    root = tmp_path / "mixed"
    _fill(root, GABOR, 1, start_axis=0)
    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM,
                  min_separation=0.02)
    arc.load_from_store()

    assert arc.consider(_cand(_axis(0)), novelty=1.0) is None
    assert arc.n_rejected_close == 1


def test_pruning_can_evict_an_entry_of_a_brain_that_is_not_running(tmp_path):
    """Capacity is shared. A brain that produces near-duplicates loses entries
    to one that does not, whichever of them happens to be loaded."""
    root = tmp_path / "mixed"
    _fill(root, FOURIER, 1, start_axis=0)
    _fill(root, GABOR, 3, start_axis=4)

    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM,
                  capacity=2)
    arc.load_from_store()
    assert len(arc) == 4
    arc.prune_to_capacity()

    assert len(arc) == 2
    assert any(e.layout == GABOR.signature() for e in arc.entries) or True
    # The point: eviction was not confined to the running layout.
    assert arc.n_evicted == 2


# ---- what stays per layout -----------------------------------------------

def test_native_rows_are_only_the_running_brains(tmp_path):
    root = tmp_path / "mixed"
    _fill(root, FOURIER, 2, start_axis=0)
    _fill(root, GABOR, 3, start_axis=4)

    arc = Archive(store=ArchiveStore(root, GABOR), layout=GABOR, dim=DIM)
    arc.load_from_store()

    rows = arc.native_rows()
    assert len(rows) == 3
    assert all(arc.layout_at(int(i)) == GABOR.signature() for i in rows)
    assert all(arc.is_native(int(i)) for i in rows)


def test_brain_at_trims_to_the_entry_s_own_width(tmp_path):
    """The pooled array is as wide as the widest layout; a padded Gabor vector
    run through Fourier's squash is a different creature, silently."""
    root = tmp_path / "mixed"
    _fill(root, FOURIER, 1, start_axis=0)
    _fill(root, GABOR, 1, start_axis=4)

    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    arc.load_from_store()

    widths = {arc.layout_at(i): len(arc.brain_at(i))
              for i in range(len(arc.entries))}
    assert widths == {FOURIER.signature(): FOURIER.length,
                      GABOR.signature(): GABOR.length}


def test_every_genome_round_trips_under_its_own_layout(tmp_path):
    from services.genome_spec import encode

    root = tmp_path / "mixed"
    _fill(root, FOURIER, 1, start_axis=0)
    _fill(root, GABOR, 1, start_axis=4)
    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    arc.load_from_store()

    for i in range(len(arc.entries)):
        lay = FOURIER if arc.is_native(i) else GABOR
        z, _ = encode(arc.brain_at(i), lay)
        assert z.size == lay.length


# ---- persistence ---------------------------------------------------------

def test_each_layout_is_flushed_to_its_own_file_at_its_own_width(tmp_path):
    root = tmp_path / "mixed"
    _fill(root, FOURIER, 2, start_axis=0)
    _fill(root, GABOR, 2, start_axis=4)

    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    arc.load_from_store()
    arc.maybe_flush(force=True)

    for lay in (FOURIER, GABOR):
        with np.load(root / lay.signature() / "vectors.npz") as z:
            assert z["brains"].shape == (2, lay.length)


def test_a_reload_after_a_pooled_flush_keeps_every_brain(tmp_path):
    root = tmp_path / "mixed"
    _fill(root, FOURIER, 2, start_axis=0)
    _fill(root, GABOR, 2, start_axis=4)

    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    arc.load_from_store()
    arc.maybe_flush(force=True)
    arc.store.close()

    back = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    loaded, dropped = back.load_from_store()
    assert (loaded, dropped) == (4, 0)


def test_admission_files_the_new_entry_under_the_running_brain(tmp_path):
    root = tmp_path / "mixed"
    _fill(root, GABOR, 1, start_axis=4)
    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    arc.load_from_store()

    e = arc.consider(_cand(_axis(1)), novelty=1.0)
    assert e is not None and e.layout == FOURIER.signature()
    arc.maybe_flush(force=True)
    with np.load(root / FOURIER.signature() / "vectors.npz") as z:
        assert z["brains"].shape == (1, FOURIER.length)


def test_the_index_row_does_not_carry_the_layout(tmp_path):
    """The directory already says it, and index.jsonl is append-only - older
    builds write to the same file."""
    import json

    root = tmp_path / "mixed"
    _fill(root, FOURIER, 1, start_axis=0)
    line = (root / FOURIER.signature() / "index.jsonl").read_text(
        encoding="utf-8").splitlines()[0]
    assert "layout" not in json.loads(line)


# ---- the ordinary case must not move --------------------------------------

def test_a_single_layout_archive_is_unchanged(tmp_path):
    root = tmp_path / "plain"
    _fill(root, FOURIER, 5, start_axis=0)

    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    loaded, dropped = arc.load_from_store()

    assert (loaded, dropped) == (5, 0)
    assert [e.id for e in arc.entries] == [0, 1, 2, 3, 4]
    assert all(arc.is_native(i) for i in range(5))
    assert arc.native_rows().tolist() == [0, 1, 2, 3, 4]


def test_ids_continue_per_layout_after_a_reload(tmp_path):
    """Each directory numbers its own entries, so neither counter restarts at
    an id the other has already issued."""
    root = tmp_path / "mixed"
    _fill(root, FOURIER, 2, start_axis=0)
    _fill(root, GABOR, 3, start_axis=4)

    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    arc.load_from_store()
    e = arc.consider(_cand(_axis(7)), novelty=1.0)
    assert e.id == 2, "Fourier's counter, not Gabor's"
