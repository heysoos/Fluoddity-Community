"""An archive must never re-issue an id it has already used.

Observed in a real archive (debug09, 2026-08-10): an unclean exit left
index.jsonl holding ids 0-1161 while vectors.npz only reached 1004. The next
two runs each reloaded, dropped the 157 unbacked entries, restarted the id
counter at 1005 and re-issued 1005-1161 - overwriting those entries'
thumbnails, because a thumbnail's filename derives from the id, and leaving
index.jsonl with duplicate ids that shadow the originals on the next load.
"""
import json

import numpy as np
import pytest

from services.archive import Archive, Candidate
from services.archive_io import ArchiveStore

DIM = 512


def _cand(seed, liveness=0.5):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, DIM).astype(np.float32)
    e /= np.linalg.norm(e)
    return Candidate(brain=rng.normal(0, 0.3, (10, 8)).astype(np.float32),
                     physics=np.zeros(8, dtype=np.float32), embedding=e,
                     liveness=liveness, spec="brain:80", viable=True)


def _fill(arc, n, start=0):
    for i in range(n):
        arc.consider(_cand(start + i), novelty=1.0, force=True)


def _ids_in_index(path):
    return [json.loads(l)["id"]
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_ids_are_not_reused_after_an_unclean_exit(tmp_path):
    """The bug, reproduced: flush at 10, admit 5 more, then die without a
    final flush. Reopening must continue from 15, not from 10."""
    root = tmp_path / "arc"
    arc = Archive(store=ArchiveStore(root))
    _fill(arc, 10)
    arc.maybe_flush(force=True)
    _fill(arc, 5, start=10)          # these reach index.jsonl but not vectors
    arc.store.close()                # no final flush: the crash

    back = Archive(store=ArchiveStore(root))
    loaded, dropped = back.load_from_store()
    assert (loaded, dropped) == (10, 5), "the trailing entries are unbacked"

    _fill(back, 3, start=100)
    ids = [e.id for e in back.entries[-3:]]
    assert ids == [15, 16, 17], f"re-issued ids that already exist: {ids}"


def test_the_index_never_gains_a_duplicate_id(tmp_path):
    """The end-to-end symptom: index.jsonl is append-only, so a reused id
    shadows the original entry on every subsequent load."""
    root = tmp_path / "arc"
    arc = Archive(store=ArchiveStore(root))
    # store.root, not root: a store lives in a subdirectory named for its brain
    # layout signature, so the files are one level down from the archive.
    store_root = arc.store.root
    _fill(arc, 10)
    arc.maybe_flush(force=True)
    _fill(arc, 5, start=10)
    arc.store.close()

    back = Archive(store=ArchiveStore(root))
    back.load_from_store()
    _fill(back, 5, start=200)
    back.maybe_flush(force=True)
    back.store.close()

    ids = _ids_in_index(store_root / "index.jsonl")
    assert len(ids) == len(set(ids)), f"duplicate ids: {len(ids) - len(set(ids))}"


def test_a_reused_id_would_overwrite_a_thumbnail(tmp_path):
    """Why the id matters beyond bookkeeping: the thumbnail filename derives
    from it, so a reused id destroys the earlier entry's picture."""
    root = tmp_path / "arc"
    arc = Archive(store=ArchiveStore(root))
    crop = np.full((32, 32, 3), 200, dtype=np.uint8)
    for i in range(6):
        arc.consider(_cand(i), novelty=1.0, force=True, thumb_crop=crop)
    arc.maybe_flush(force=True)
    for i in range(4):
        arc.consider(_cand(50 + i), novelty=1.0, force=True, thumb_crop=crop)
    # store.root, not root - see test_the_index_never_gains_a_duplicate_id.
    # Globbing the archive directory finds nothing and the assertion below
    # passes without testing anything.
    doomed = sorted(p.name for p in (arc.store.root / "thumbs").glob("*.jpg"))[-4:]
    assert doomed, "no thumbnails were written, so this proves nothing"
    arc.store.close()

    back = Archive(store=ArchiveStore(root))
    back.load_from_store()
    for i in range(4):
        back.consider(_cand(90 + i), novelty=1.0, force=True, thumb_crop=crop)

    written = {e.thumb for e in back.entries[-4:]}
    assert not (written & set(doomed)), "a new entry claimed an existing thumbnail"


def test_eviction_also_must_not_free_an_id_for_reuse(tmp_path):
    """Not only crashes: prune_to_capacity can remove the highest-id entry, so
    max(survivors) is not max(issued) even after a perfectly clean quit."""
    root = tmp_path / "arc"
    arc = Archive(store=ArchiveStore(root))
    _fill(arc, 8)
    # Evict the newest, which is what makes the survivor max lag the issued max.
    arc._remove(len(arc.entries) - 1)
    arc.maybe_flush(force=True)
    arc.store.close()

    back = Archive(store=ArchiveStore(root))
    back.load_from_store()
    _fill(back, 1, start=300)
    assert back.entries[-1].id == 8, "an evicted entry's id came back"


def test_a_clean_round_trip_still_numbers_consecutively(tmp_path):
    """The fix must not leave gaps in the ordinary case."""
    root = tmp_path / "arc"
    arc = Archive(store=ArchiveStore(root))
    _fill(arc, 5)
    arc.maybe_flush(force=True)
    arc.store.close()

    back = Archive(store=ArchiveStore(root))
    back.load_from_store()
    _fill(back, 2, start=400)
    assert [e.id for e in back.entries] == [0, 1, 2, 3, 4, 5, 6]


def test_an_empty_archive_starts_at_zero(tmp_path):
    back = Archive(store=ArchiveStore(tmp_path / "arc"))
    back.load_from_store()
    _fill(back, 1)
    assert back.entries[0].id == 0
