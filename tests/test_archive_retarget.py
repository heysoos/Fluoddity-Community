"""Re-pointing an archive at another brain, without reloading it.

Every layout's entries are already in memory - load_from_store reads every
signature directory - so a layout change inside one archive changes only which
rows are NATIVE. Everything else in an archive is about PICTURES.
"""
from __future__ import annotations

import numpy as np

from services.archive import Archive, Candidate
from services.archive_io import ArchiveStore
from services.brains import BrainLayout, default_layout

DIM = 8
FOURIER = default_layout()
GABOR = BrainLayout("gabor", (12,), 168)
WIDE = BrainLayout("gabor", (25,), 350)


def _cand(vec, brain_len):
    e = np.asarray(vec, dtype=np.float32)
    e = e / max(float(np.linalg.norm(e)), 1e-8)
    return Candidate(embedding=e, brain=np.arange(brain_len, dtype=np.float32),
                     physics=np.zeros(8, np.float32), liveness=0.5,
                     spec="brain", goal="", run_id="r", gen=0, tile=0,
                     viable=True)


def _axis(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def _fill(root, layout, n, start_axis):
    store = ArchiveStore(root, layout)
    arc = Archive(store=store, layout=layout, dim=DIM)
    for j in range(n):
        arc.consider(_cand(_axis(start_axis + j), layout.length),
                     novelty=1.0, force=True)
    arc.maybe_flush(force=True)
    store.close()


def _mixed(tmp_path):
    """An archive holding both brains, opened under Fourier."""
    root = tmp_path / "mixed"
    _fill(root, FOURIER, 2, 0)
    _fill(root, GABOR, 3, 4)
    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    arc.load_from_store()
    return root, arc


def _snapshot(arc):
    """Everything a retarget must leave alone."""
    return {
        "ids": [e.id for e in arc.entries],
        "layouts": [arc.layout_at(i) for i in range(len(arc.entries))],
        "emb": arc.embeddings.copy(),
        "novelty": np.array([e.novelty for e in arc.entries], dtype=np.float64),
        "brains": [arc.brain_at(i).copy() for i in range(len(arc.entries))],
    }


def test_retarget_moves_only_which_rows_are_native(tmp_path):
    _root, arc = _mixed(tmp_path)
    before = _snapshot(arc)
    assert len(arc.native_rows()) == 2

    arc.retarget(GABOR)

    assert arc.signature == GABOR.signature()
    assert len(arc.native_rows()) == 3
    after = _snapshot(arc)
    assert after["ids"] == before["ids"]
    assert after["layouts"] == before["layouts"]
    np.testing.assert_array_equal(after["emb"], before["emb"])
    np.testing.assert_array_equal(after["novelty"], before["novelty"])
    for a, b in zip(after["brains"], before["brains"]):
        np.testing.assert_array_equal(a, b)


def _native_entries(arc):
    """Which ENTRIES the running brain can decode, named by (signature, id).

    Not by row index: load_from_store puts the running layout FIRST, so a
    reload orders the rows differently from a retarget - and reordering is the
    array rebuild a retarget exists to avoid. An id is unique only within its
    own signature directory, so both halves are needed to name one.
    """
    return sorted((arc.layout_at(int(i)), arc.entries[int(i)].id)
                  for i in arc.native_rows())


def test_retarget_agrees_with_release_and_reload(tmp_path):
    """The load-bearing case: the fast path must be indistinguishable from the
    teardown it replaces."""
    root, arc = _mixed(tmp_path)
    arc.retarget(GABOR)
    fast = _snapshot(arc)
    fast_native = _native_entries(arc)

    slow_arc = Archive(store=ArchiveStore(root, GABOR), layout=GABOR, dim=DIM)
    slow_arc.load_from_store()
    slow = _snapshot(slow_arc)
    slow_native = _native_entries(slow_arc)

    assert sorted(fast["ids"]) == sorted(slow["ids"])
    assert fast_native == slow_native
    assert sorted(fast["layouts"]) == sorted(slow["layouts"])


def test_retarget_does_not_dirty_the_novelty_column(tmp_path):
    """Nothing was admitted and nothing removed, so every entry is still
    scored against exactly the set now held. Dirtying it would buy a full
    rescore per layout move for no information."""
    _root, arc = _mixed(tmp_path)
    arc.rescore_all()
    assert arc._novelty_clean is True
    arc.retarget(GABOR)
    assert arc._novelty_clean is True


def test_retarget_to_a_layout_the_archive_has_never_held(tmp_path):
    """A brand new brain has no directory yet, and admitting under it must
    still find a store to write through."""
    _root, arc = _mixed(tmp_path)
    arc.retarget(WIDE)
    assert arc.signature == WIDE.signature()
    assert len(arc.native_rows()) == 0
    assert arc.stores.get(WIDE.signature()) is not None


def test_retarget_to_a_wider_layout_pads_without_reinterpreting(tmp_path):
    """brain_at() trims each row back to its own width, so widening the pooled
    column must leave every stored genome reading exactly as before."""
    _root, arc = _mixed(tmp_path)
    before = [arc.brain_at(i).copy() for i in range(len(arc.entries))]
    arc.retarget(WIDE)
    for i, b in enumerate(before):
        np.testing.assert_array_equal(arc.brain_at(i), b)


def test_retarget_is_visible_to_the_thumbnail_loader(tmp_path):
    """gl_loader closes over the LIVE stores dict, so a store added by a
    retarget has to be reachable through it without rebuilding the cache."""
    _root, arc = _mixed(tmp_path)
    stores = arc.stores
    arc.retarget(WIDE)
    assert stores.get(WIDE.signature()) is not None
