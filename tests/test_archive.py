import numpy as np
import pytest

from services.archive import AdaptiveThreshold, Archive, Candidate
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
    kw.setdefault("seed_n", 0)      # novelty gate on from the first candidate
    kw.setdefault("capacity", 100)
    return Archive(store=None, **kw)


# ---- adaptive threshold ------------------------------------------------

def test_threshold_rises_when_admission_is_too_generous():
    t = AdaptiveThreshold(initial=0.1, target_rate=0.15, window=10)
    for _ in range(50):
        t.observe(True)
    assert t.value > 0.1


def test_threshold_falls_when_nothing_gets_in():
    t = AdaptiveThreshold(initial=0.1, target_rate=0.15, window=10)
    for _ in range(50):
        t.observe(False)
    assert t.value < 0.1


def test_threshold_converges_toward_the_target_rate():
    rng = np.random.default_rng(0)
    t = AdaptiveThreshold(initial=0.5, target_rate=0.15, window=100)
    for _ in range(3000):
        t.observe(bool(rng.random() < 0.15))
    assert t.rate == pytest.approx(0.15, abs=0.08)


def test_threshold_is_clamped_to_its_bounds():
    t = AdaptiveThreshold(initial=0.5, target_rate=0.15, window=5, lo=0.01, hi=0.6)
    for _ in range(500):
        t.observe(True)
    assert t.value <= 0.6
    for _ in range(2000):
        t.observe(False)
    assert t.value >= 0.01


# ---- the three gates ---------------------------------------------------

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


def test_the_novelty_gate_rejects_a_near_duplicate():
    a = fresh(liveness_min=0.0)
    a.threshold.value = 0.5
    assert a.consider(cand([1, 0, 0, 0]), novelty=0.4) is None
    assert len(a) == 0


def test_a_non_finite_descriptor_is_rejected_and_counted():
    a = fresh()
    c = cand([1, 0, 0, 0])
    c.embedding = np.array([np.nan, 0, 0, 0], dtype=np.float32)
    assert a.consider(c, novelty=1.0) is None
    assert a.stats()["n_nonfinite"] == 1


def test_each_gate_fires_independently():
    """Failing one gate must not depend on the others passing."""
    a = fresh(liveness_min=0.5)
    a.threshold.value = 0.5
    assert a.consider(cand([1, 0, 0, 0], liveness=1.0, viable=False), 1.0) is None
    assert a.consider(cand([0, 1, 0, 0], liveness=0.1), 1.0) is None
    assert a.consider(cand([0, 0, 1, 0], liveness=1.0), 0.1) is None
    assert a.consider(cand([0, 0, 0, 1], liveness=1.0), 1.0) is not None


def test_rejected_descriptors_go_into_the_rejects_ring():
    a = fresh(liveness_min=0.5)
    a.consider(cand([1, 0, 0, 0], liveness=0.0), novelty=1.0)
    assert len(a.rejects) == 1


# ---- bootstrap ---------------------------------------------------------

def test_bootstrap_skips_only_the_novelty_gate():
    a = fresh(seed_n=5, liveness_min=0.02)
    a.threshold.value = 0.9
    assert a.consider(cand([1, 0, 0, 0]), novelty=0.0) is not None, "novelty off"
    assert a.consider(cand([0, 1, 0, 0], liveness=0.0), novelty=1.0) is None, "liveness on"
    assert a.consider(cand([0, 0, 1, 0], viable=False), novelty=1.0) is None, "viability on"


def test_the_novelty_gate_switches_on_at_seed_n():
    a = fresh(seed_n=2, liveness_min=0.0)
    a.threshold.value = 0.9
    a.consider(cand([1, 0, 0, 0]), novelty=0.0)
    a.consider(cand([0, 1, 0, 0]), novelty=0.0)
    assert len(a) == 2
    assert a.consider(cand([0, 0, 1, 0]), novelty=0.0) is None


def test_the_threshold_is_not_updated_during_bootstrap():
    """A cold start would drive the threshold on evidence that means nothing."""
    a = fresh(seed_n=10, liveness_min=0.0)
    before = a.threshold.value
    for i in range(5):
        a.consider(cand([float(i), 1, 0, 0]), novelty=0.0)
    assert a.threshold.value == before


# ---- pins --------------------------------------------------------------

def test_a_pin_bypasses_every_gate():
    a = fresh(liveness_min=0.9)
    a.threshold.value = 0.99
    e = a.consider(cand([1, 0, 0, 0], liveness=0.0, viable=False), 0.0, pinned=True)
    assert e is not None
    assert e.pinned is True
    assert e.source == "pin"


def test_pins_survive_eviction():
    a = fresh(capacity=3, liveness_min=0.0, seed_n=0)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=0.0, pinned=True)
    for i in range(10):
        a.consider(cand([0, float(i + 1), 0, 0]), novelty=0.9)
    assert len(a) == 3
    assert any(e.pinned for e in a.entries)


# ---- capacity ----------------------------------------------------------

def test_eviction_drops_the_least_novel_entry():
    a = fresh(capacity=2, liveness_min=0.0)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=0.9)
    a.consider(cand([0, 1, 0, 0]), novelty=0.1)      # the redundant one
    a.consider(cand([0, 0, 1, 0]), novelty=0.8)
    assert len(a) == 2
    assert sorted(round(e.novelty, 1) for e in a.entries) == [0.8, 0.9]


def test_eviction_keeps_arrays_and_entries_in_lockstep():
    a = fresh(capacity=3, liveness_min=0.0)
    a.threshold.value = 0.0
    for i in range(8):
        a.consider(cand([float(i), 0, 0, 0]), novelty=float(i) / 10.0)
    assert len(a) == 3
    assert a.embeddings.shape == (3, 4)
    assert a.brains.shape == (3, 10, 8)
    assert a.physics.shape == (3, 8)
    assert len(a.entries) == 3


def test_an_all_pinned_archive_at_capacity_stops_admitting():
    a = fresh(capacity=2, liveness_min=0.0)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), 0.0, pinned=True)
    a.consider(cand([0, 1, 0, 0]), 0.0, pinned=True)
    a.consider(cand([0, 0, 1, 0]), novelty=0.9)
    assert len(a) == 2
    assert a.stats()["blocked_by_pins"] is True


# ---- novelty bookkeeping ----------------------------------------------

def test_novelty_of_uses_both_the_archive_and_the_rejects_ring():
    a = fresh(liveness_min=0.0, k=1)
    a.threshold.value = 0.0
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
    a.threshold.value = 0.0
    for i in range(4):
        a.consider(cand([1.0, 0.05 * i, 0.0, 0.0]), novelty=1.0)
    assert all(e.novelty == 1.0 for e in a.entries), "admitted at the stale value"
    assert a.refresh(4) == 4
    assert all(e.novelty < 0.5 for e in a.entries), "novelty must have dropped"


def test_refresh_excludes_the_entry_from_its_own_neighbours():
    """Without exclude_self every entry's nearest neighbour is itself at
    distance 0, and every novelty collapses toward zero."""
    a = fresh(liveness_min=0.0, k=1)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([0, 1, 0, 0]), novelty=1.0)
    a.refresh(2)
    assert all(e.novelty == pytest.approx(1.0, abs=1e-5) for e in a.entries)


def test_refresh_walks_round_robin():
    a = fresh(liveness_min=0.0)
    a.threshold.value = 0.0
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
    a.threshold.value = 0.0
    assert a.centroid() is None
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([0, 1, 0, 0]), novelty=1.0)
    c = a.centroid()
    assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-5)


def test_nearest_returns_the_best_match_for_a_goal():
    a = fresh(liveness_min=0.0)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([0, 1, 0, 0]), novelty=1.0)
    goal = _unit(np.array([[0.1, 1.0, 0, 0]], np.float32))[0]
    assert a.nearest(goal) == 1
    assert fresh().nearest(goal) is None


# ---- persistence -------------------------------------------------------

def _seeded_store_archive(tmp_path, n=3):
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=4, seed_n=0, liveness_min=0.0, capacity=100)
    a.threshold.value = 0.0
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
    a = Archive(store=store, dim=4, seed_n=0, liveness_min=0.0, capacity=100)
    a.threshold.value = 0.0
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

    b = Archive(store=ArchiveStore(tmp_path), dim=4, seed_n=0, liveness_min=0.0)
    b.load_from_store()
    b.threshold.value = 0.0
    e = b.consider(cand([0, 0, 0, 1]), novelty=1.0)
    assert e.id == 3, "ids must not collide with the reloaded entries"


def test_maybe_flush_only_writes_on_the_interval(tmp_path):
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=4, seed_n=0, liveness_min=0.0, capacity=100)
    a.threshold.value = 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    assert a.maybe_flush(every=5) is False
    assert a.maybe_flush(every=5, force=True) is True
    store.close()


def test_an_archive_with_no_store_works_entirely_in_memory():
    a = fresh()
    assert a.consider(cand([1, 0, 0, 0]), novelty=1.0) is not None
    assert a.maybe_flush(force=True) is False
    assert a.load_from_store() == (0, 0)
