import numpy as np
import pytest

from services.archive import AdmissionRate, Archive, Candidate
from services.archive_io import ArchiveStore


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def cand(vec, liveness=1.0, viable=True, dim=4, spec="brain:80"):
    e = np.zeros(dim, dtype=np.float32)
    e[: len(vec)] = vec
    return Candidate(
        brain=np.zeros((10, 8), np.float32),
        physics=np.zeros(8, np.float32),
        embedding=_unit(e[None])[0],
        liveness=liveness,
        viable=viable,
        spec=spec,
    )


def fresh(**kw):
    kw.setdefault("dim", 4)
    kw.setdefault("capacity", 100)
    return Archive(store=None, **kw)


# ---- admission rate is a readout, never a controller -------------------

def test_the_admission_rate_reports_the_recent_fraction():
    r = AdmissionRate(window=10)
    for i in range(10):
        r.observe(i < 3)
    assert r.rate == pytest.approx(0.3)


def test_the_admission_rate_forgets_beyond_its_window():
    r = AdmissionRate(window=5)
    for _ in range(5):
        r.observe(False)
    for _ in range(5):
        r.observe(True)
    assert r.rate == pytest.approx(1.0)


def test_an_unobserved_admission_rate_is_zero():
    assert AdmissionRate().rate == 0.0


def test_the_admission_rate_has_nothing_to_steer():
    """The regression guard for the whole change.

    Its predecessor multiplied a novelty threshold by 1.05 or 0.95 on EVERY
    observation, 16 to 64 times a generation, against a rate averaged over six
    generations. Anything here that a candidate can move is a control loop
    waiting to happen, so there must be nothing but the window.
    """
    r = AdmissionRate()
    tunables = {k for k in vars(r) if not k.startswith("_")}
    assert tunables == {"window"}


# ---- the gates ---------------------------------------------------------

def test_a_novel_live_viable_candidate_is_admitted():
    a = fresh()
    e = a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    assert e is not None
    assert len(a) == 1
    assert e.source == "expansion"


def test_the_viability_gate_rejects_a_dead_tile():
    a = fresh()
    assert a.consider(cand([1, 0, 0, 0], viable=False), novelty=1.0) is None
    assert len(a) == 0


def test_the_liveness_gate_rejects_a_frozen_pattern():
    a = fresh(liveness_min=0.02)
    assert a.consider(cand([1, 0, 0, 0], liveness=0.0), novelty=1.0) is None
    assert len(a) == 0


def test_novelty_does_not_gate_admission():
    """The point of the change: a near-duplicate still gets in.

    It is kept until something more novel needs its slot, which is the reverse
    of the old bargain - rejecting cost a full 2000-step rollout permanently,
    admitting costs one slot.
    """
    a = fresh(liveness_min=0.0)
    assert a.consider(cand([1, 0, 0, 0]), novelty=0.0) is not None
    assert len(a) == 1


def test_nothing_a_candidate_carries_can_change_a_later_verdict():
    """No hidden state between candidates - no controller left to wind up."""
    a = fresh(liveness_min=0.0)
    for i in range(200):
        a.consider(cand([float(i), 1, 0, 0]), novelty=0.0)
    assert len(a) == 200, "200 identical-novelty candidates, 200 admissions"


def test_a_non_finite_descriptor_is_rejected_and_counted():
    a = fresh()
    c = cand([1, 0, 0, 0])
    c.embedding = np.array([np.nan, 0, 0, 0], dtype=np.float32)
    assert a.consider(c, novelty=1.0) is None
    assert a.stats()["n_nonfinite"] == 1


def test_each_gate_fires_independently():
    """Failing one gate must not depend on the other passing."""
    a = fresh(liveness_min=0.5)
    assert a.consider(cand([1, 0, 0, 0], liveness=1.0, viable=False), 1.0) is None
    assert a.consider(cand([0, 1, 0, 0], liveness=0.1), 1.0) is None
    assert a.consider(cand([0, 0, 1, 0], liveness=1.0), 0.1) is not None, (
        "low novelty is not a gate")
    assert a.consider(cand([0, 0, 0, 1], liveness=1.0), 1.0) is not None


def test_rejected_descriptors_go_into_the_rejects_ring():
    a = fresh(liveness_min=0.5)
    a.consider(cand([1, 0, 0, 0], liveness=0.0), novelty=1.0)
    assert len(a.rejects) == 1


# ---- admission does not depend on how full the archive is --------------

def test_admission_is_the_same_cold_and_warm():
    """There is no bootstrap phase left to special-case.

    seed_n used to hold the novelty gate off until the archive had 256 entries,
    so the FIRST candidates and the 300th obeyed different rules. Archive no
    longer knows about seed_n at all - the driver keeps it, to choose bootstrap
    vs expansion, which is a question about the search rather than admission.
    """
    a = fresh(liveness_min=0.02)
    assert a.consider(cand([1, 0, 0, 0]), novelty=0.0) is not None
    for i in range(300):
        a.consider(cand([float(i), 1, 0, 0]), novelty=0.0)
    assert a.consider(cand([0, 0, 1, 0]), novelty=0.0) is not None
    assert a.consider(cand([0, 1, 0, 0], liveness=0.0), novelty=1.0) is None
    assert a.consider(cand([0, 0, 1, 0], viable=False), novelty=1.0) is None


def test_the_archive_no_longer_takes_a_seed_n():
    import inspect

    assert "seed_n" not in inspect.signature(Archive).parameters


# ---- pins --------------------------------------------------------------

def test_a_pin_bypasses_every_gate():
    a = fresh(liveness_min=0.9)
    e = a.consider(cand([1, 0, 0, 0], liveness=0.0, viable=False), 0.0, pinned=True)
    assert e is not None
    assert e.pinned is True
    assert e.source == "pin"


def test_pins_survive_eviction():
    a = fresh(capacity=3, liveness_min=0.0)
    a.consider(cand([1, 0, 0, 0]), novelty=0.0, pinned=True)
    for i in range(10):
        a.consider(cand([0, float(i + 1), 0, 0]), novelty=0.9)
    a.prune_to_capacity()
    assert len(a) == 3
    assert any(e.pinned for e in a.entries), (
        "the pin has the lowest novelty here and must still survive")


# ---- capacity is the pruning rule --------------------------------------

def test_pruning_drops_the_least_novel_entries():
    a = fresh(capacity=2, liveness_min=0.0)
    a.consider(cand([1, 0, 0, 0]), novelty=0.9)
    a.consider(cand([0, 1, 0, 0]), novelty=0.1)      # the redundant one
    a.consider(cand([0, 0, 1, 0]), novelty=0.8)
    assert len(a) == 3, "all three are admitted; pruning happens afterwards"
    assert a.prune_to_capacity() == 1
    assert sorted(round(e.novelty, 1) for e in a.entries) == [0.8, 0.9]


def test_pruning_below_capacity_does_nothing():
    a = fresh(capacity=10, liveness_min=0.0)
    a.consider(cand([1, 0, 0, 0]), novelty=0.1)
    assert a.prune_to_capacity() == 0
    assert len(a) == 1


def test_pruning_takes_a_whole_generations_worth_in_one_pass():
    a = fresh(capacity=100, liveness_min=0.0)
    for i, v in enumerate(_cone(np.random.default_rng(7), 164, 4)):
        a.consider(cand(v.tolist()), novelty=float(i) / 164.0)
    assert a.prune_to_capacity() == 64
    assert len(a) == 100
    assert min(e.novelty for e in a.entries) == pytest.approx(64 / 164, abs=1e-6)


def test_pruning_keeps_arrays_and_entries_in_lockstep():
    a = fresh(capacity=3, liveness_min=0.0)
    for i in range(8):
        a.consider(cand([float(i), 0, 0, 0]), novelty=float(i) / 10.0)
    a.prune_to_capacity()
    assert len(a) == 3
    assert a.embeddings.shape == (3, 4)
    assert a.brains.shape == (3, 80)   # flat and layout-wide, not Fourier (10, 8)
    assert a.physics.shape == (3, 8)
    assert len(a.entries) == 3


def test_pruning_removes_exactly_the_intended_entries():
    """_remove swaps the LAST entry into the hole, so removing low-index-first
    would relocate an entry still queued for removal. Ids pin the identity."""
    a = fresh(capacity=4, liveness_min=0.0)
    for i in range(12):
        a.consider(cand([float(i), 1, 0, 0]), novelty=float(i))
    a.prune_to_capacity()
    assert sorted(e.novelty for e in a.entries) == [8.0, 9.0, 10.0, 11.0]
    assert sorted(e.id for e in a.entries) == [8, 9, 10, 11]


def test_an_all_pinned_archive_over_capacity_cannot_be_pruned():
    a = fresh(capacity=2, liveness_min=0.0)
    a.consider(cand([1, 0, 0, 0]), 0.0, pinned=True)
    a.consider(cand([0, 1, 0, 0]), 0.0, pinned=True)
    a.consider(cand([0, 0, 1, 0]), novelty=0.9)
    assert a.prune_to_capacity() == 1, "the one unpinned entry goes"
    a.consider(cand([0, 0, 0, 1]), 0.0, pinned=True)
    assert a.prune_to_capacity() == 0
    assert len(a) == 3, "nothing evictable, so it grows past capacity"
    assert a.stats()["blocked_by_pins"] is True


def test_eviction_deletes_the_thumbnail(tmp_path):
    """Otherwise a full archive at grid 8 orphans 64 JPEGs every ~2.8 s."""
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=4, liveness_min=0.0, capacity=1)
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    doomed = a.consider(cand([1, 0, 0, 0]), novelty=0.1, thumb_crop=crop)
    a.consider(cand([0, 1, 0, 0]), novelty=0.9, thumb_crop=crop)
    assert store.thumb_path(doomed.thumb).is_file()

    assert a.prune_to_capacity() == 1
    assert not store.thumb_path(doomed.thumb).exists()
    assert len(list((store.root / "thumbs").iterdir())) == 1
    store.close()


# ---- novelty bookkeeping ----------------------------------------------

def test_novelty_of_uses_both_the_archive_and_the_rejects_ring():
    a = fresh(liveness_min=0.0, k=1)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    far = _unit(np.array([[0, 0, 0, 1]], np.float32))
    before = float(a.novelty_of(far)[0])
    a.rejects.add(far)
    after = float(a.novelty_of(far)[0])
    assert after < before, "a recent rejection must suppress novelty there"


def test_novelty_of_an_empty_archive_is_one():
    a = fresh()
    assert a.novelty_of(_unit(np.eye(4)[:2]))[0] == pytest.approx(1.0)


def test_refresh_rescores_entries_against_the_current_archive():
    """Entries are CLUSTERED, not orthogonal: an orthonormal set sits at
    distance exactly 1.0 from every neighbour, so refreshing it cannot lower
    anything and the test would prove nothing."""
    a = fresh(liveness_min=0.0, k=1)
    for i in range(4):
        a.consider(cand([1.0, 0.05 * i, 0.0, 0.0]), novelty=1.0)
    assert all(e.novelty == 1.0 for e in a.entries), "admitted at the stale value"
    assert a.refresh(4) == 4
    assert all(e.novelty < 0.5 for e in a.entries), "novelty must have dropped"


def test_refresh_excludes_the_entry_from_its_own_neighbours():
    """Without exclude_self every entry's nearest neighbour is itself at
    distance 0, and every novelty collapses toward zero."""
    a = fresh(liveness_min=0.0, k=1)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([0, 1, 0, 0]), novelty=1.0)
    a.refresh(2)
    assert all(e.novelty == pytest.approx(1.0, abs=1e-5) for e in a.entries)


def test_refresh_walks_round_robin():
    a = fresh(liveness_min=0.0)
    for i in range(6):
        v = np.zeros(4)
        v[i % 4] = float(i + 1)
        a.consider(cand(v.tolist()), novelty=1.0)
    assert a.refresh(2) == 2
    assert a.refresh(2) == 2
    assert a._refresh_cursor == 4


def test_refresh_on_an_empty_archive_is_a_no_op():
    assert fresh().refresh(10) == 0


def test_centroid_is_unit_norm_and_none_when_empty():
    a = fresh(liveness_min=0.0)
    assert a.centroid() is None
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([0, 1, 0, 0]), novelty=1.0)
    c = a.centroid()
    assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-5)


def test_nearest_returns_the_best_match_for_a_goal():
    a = fresh(liveness_min=0.0)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([0, 1, 0, 0]), novelty=1.0)
    goal = _unit(np.array([[0.1, 1.0, 0, 0]], np.float32))[0]
    assert a.nearest(goal) == 1
    assert fresh().nearest(goal) is None


# ---- persistence -------------------------------------------------------

def _seeded_store_archive(tmp_path, n=3):
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=4, liveness_min=0.0, capacity=100)
    for i in range(n):
        v = np.zeros(4)
        v[i % 4] = 1.0
        a.consider(cand(v.tolist()), novelty=1.0)
    return store, a


def test_load_from_store_reconciles_index_against_vectors(tmp_path):
    store, a = _seeded_store_archive(tmp_path)
    a.maybe_flush(force=True)
    store.close()

    b = Archive(store=ArchiveStore(tmp_path), dim=4)
    loaded, dropped = b.load_from_store()
    assert loaded == 3
    assert dropped == 0
    assert len(b) == 3
    assert [e.id for e in b.entries] == [0, 1, 2]


def test_entries_with_no_matching_vector_row_are_dropped(tmp_path):
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=4, liveness_min=0.0, capacity=100)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.maybe_flush(force=True)
    # a second entry reaches the index but the process dies before the flush
    a.consider(cand([0, 1, 0, 0]), novelty=1.0)
    store.close()

    b = Archive(store=ArchiveStore(tmp_path), dim=4)
    loaded, dropped = b.load_from_store()
    assert loaded == 1
    assert dropped == 1


def test_next_id_continues_after_a_reload(tmp_path):
    store, a = _seeded_store_archive(tmp_path)
    a.maybe_flush(force=True)
    store.close()

    b = Archive(store=ArchiveStore(tmp_path), dim=4, liveness_min=0.0)
    b.load_from_store()
    e = b.consider(cand([0, 0, 0, 1]), novelty=1.0)
    assert e.id == 3, "ids must not collide with the reloaded entries"


def test_maybe_flush_only_writes_on_the_interval(tmp_path):
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=4, liveness_min=0.0, capacity=100)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    assert a.maybe_flush(every=5) is False
    assert a.maybe_flush(every=5, force=True) is True
    store.close()


def test_an_archive_with_no_store_works_entirely_in_memory():
    a = fresh()
    assert a.consider(cand([1, 0, 0, 0]), novelty=1.0) is not None
    assert a.maybe_flush(force=True) is False
    assert a.load_from_store() == (0, 0)


# ---- novelty is a LIVE column, not an at-admission stamp ----------------

def _cone(rng, n, dim, spread=0.35):
    """Vectors in a narrow cone, like real CLIP descriptors.

    Independent gaussian directions in a low dimension are near-ORTHOGONAL, so
    every kNN distance is ~1.0 and a stale 1.0 is indistinguishable from a
    correct rescore - which makes such a fixture prove nothing here. Measured
    over 4784 real descriptors the mean pairwise cosine is 0.897; this puts the
    fixture in the same regime.
    """
    base = np.zeros(dim, dtype=np.float64)
    base[0] = 1.0
    return base + spread * rng.normal(size=(n, dim))


def _scattered_store(tmp_path, n=40, dim=8, stamped=6):
    """An archive whose first `stamped` entries carry novelty 1.0.

    That is what a real generation 0 looks like: novelty_from_distances returns
    1.0 by convention when there is no reference yet, so every tile of the
    first generation is stamped maximally novel and index.jsonl keeps it.
    """
    rng = np.random.default_rng(4)
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=dim, liveness_min=0.0, capacity=1000)
    for i, v in enumerate(_cone(rng, n, dim)):
        a.consider(cand(v.tolist(), dim=dim), novelty=1.0 if i < stamped else 0.03)
    return store, a


def _parent_weights(archive, alpha=4.0):
    """p ~ novelty^alpha - exactly what _ask_expansion and latent_goal use."""
    w = np.clip(np.array([e.novelty for e in archive.entries], np.float64), 0, None)
    w = w ** alpha
    return w / w.sum()


def test_rescore_all_replaces_every_stored_novelty():
    a = fresh(dim=8, capacity=1000)
    for v in _cone(np.random.default_rng(1), 30, 8):
        a.consider(cand(v.tolist(), dim=8), novelty=1.0)
    assert all(e.novelty == 1.0 for e in a.entries)

    assert a.rescore_all() == 30
    assert all(0.0 < e.novelty < 0.9 for e in a.entries), (
        "a real kNN distance inside a cone cannot be the no-reference 1.0")


def test_rescore_all_on_an_empty_archive_is_a_no_op():
    assert fresh().rescore_all() == 0


def test_a_reload_does_not_inherit_the_at_admission_novelty(tmp_path):
    store, a = _scattered_store(tmp_path)
    a.maybe_flush(force=True)
    store.close()

    b = Archive(store=ArchiveStore(tmp_path), dim=8)
    b.load_from_store()
    assert len(b) == 40
    assert not any(e.novelty >= 0.999 for e in b.entries), (
        "generation 0's 1.0 stamp must not survive a reload")


def test_a_reload_does_not_hand_generation_zero_the_whole_parent_weight(tmp_path):
    """The defect this whole change exists for.

    novelty^4 turns a stamped 1.0 against a typical 0.03 into a million-to-one
    weight, so on the default archive 100.0% of parent-selection probability
    landed on the 58 entries of generation 0 - one of which is a black frame.
    """
    store, a = _scattered_store(tmp_path, stamped=6)
    assert _parent_weights(a)[:6].sum() > 0.9999, (
        "precondition: as stored, the stamped entries take everything")
    a.maybe_flush(force=True)
    store.close()

    b = Archive(store=ArchiveStore(tmp_path), dim=8)
    b.load_from_store()
    # 6 of 40 entries is 15% of the archive; they may still be genuinely novel,
    # but they cannot own the distribution.
    assert _parent_weights(b)[:6].sum() < 0.6


def test_a_flush_writes_the_current_novelty_not_the_admitted_one(tmp_path):
    store, a = _scattered_store(tmp_path, n=20, stamped=20)
    a.rescore_all()
    live = [e.novelty for e in a.entries]
    a.maybe_flush(force=True)
    store.close()

    with np.load(store.vectors_path, allow_pickle=False) as z:
        assert "novelty" in z.files
        assert z["novelty"] == pytest.approx(np.float32(live), abs=1e-6)


def test_an_archive_saved_before_novelty_was_persisted_still_loads(tmp_path):
    """vectors.npz gained the array on 2026-08-08. Older files must open."""
    store, a = _scattered_store(tmp_path, n=12, stamped=12)
    a.maybe_flush(force=True)
    store.close()

    with np.load(store.vectors_path, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files if k != "novelty"}
    np.savez(store.vectors_path, **old)

    b = Archive(store=ArchiveStore(tmp_path), dim=8)
    loaded, dropped = b.load_from_store()
    assert (loaded, dropped) == (12, 0)
    assert not store.vectors_path.with_suffix(".npz.bad").exists()
    assert not any(e.novelty >= 0.999 for e in b.entries)


# ---- revision ------------------------------------------------------------
#
# The browser derives a PCA projection and a sort order from the archive, both
# O(n). Without a version to compare against, the only safe assumption is that
# they are stale, so they were recomputed 60 times a second - 40 ms a frame at
# the 20000 capacity. Every mutation below has to move it or the map freezes.


def test_admission_moves_the_revision():
    a = fresh(capacity=10, liveness_min=0.0)
    before = a.revision
    a.consider(cand([1, 0, 0, 0]), 1.0)
    assert a.revision > before


def test_a_rejected_candidate_does_not_move_the_revision():
    """Nothing a viewer draws changed, so nothing it caches need be thrown
    away."""
    a = fresh(capacity=10, liveness_min=0.5)
    a.consider(cand([1, 0, 0, 0]), 1.0)
    before = a.revision
    assert a.consider(cand([0, 1, 0, 0], liveness=0.01), 1.0) is None
    assert a.revision == before


def test_eviction_moves_the_revision():
    a = fresh(capacity=2, liveness_min=0.0)
    for i in range(3):
        a.consider(cand([float(i), 1, 0, 0]), 1.0 - 0.1 * i)
    before = a.revision
    assert a.prune_to_capacity() == 1
    assert a.revision > before


def test_rescoring_moves_the_revision():
    """Novelty is a displayed column and the gallery sorts on it, so a sweep
    that leaves the revision alone shows the old order."""
    a = fresh(capacity=10, liveness_min=0.0)
    for i in range(4):
        a.consider(cand([float(i), 1, 0, 0]), 1.0)
    before = a.revision
    a.rescore_all()
    assert a.revision > before
    partial = a.revision
    a.refresh(2)
    assert a.revision > partial
