import numpy as np
import pytest

from services.novelty import (
    RejectsRing,
    knn_distances,
    knn_novelty,
    novelty_from_distances,
    sample_by_novelty,
)


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def test_knn_distances_are_sorted_ascending():
    ref = _unit(np.eye(6))
    q = _unit(np.array([[1.0, 0.2, 0.0, 0.0, 0.0, 0.0]]))
    d = knn_distances(q, ref, k=4)
    assert d.shape == (1, 4)
    assert np.all(np.diff(d[0]) >= -1e-6)


def test_knn_distances_match_a_brute_force_reference():
    rng = np.random.default_rng(3)
    ref = _unit(rng.normal(size=(50, 8)))
    q = _unit(rng.normal(size=(7, 8)))
    got = knn_distances(q, ref, k=5)
    full = np.sort(1.0 - (q @ ref.T), axis=1)[:, :5]
    assert np.allclose(got, full, atol=1e-6)


def test_knn_distances_clamps_k_to_the_reference_size():
    ref = _unit(np.eye(3))
    assert knn_distances(_unit(np.eye(3)[:1]), ref, k=10).shape == (1, 3)


def test_knn_distances_on_an_empty_reference_returns_no_columns():
    assert knn_distances(_unit(np.eye(2)), np.zeros((0, 2), np.float32), k=5).shape == (2, 0)


def test_exclude_self_drops_the_zero_distance():
    ref = _unit(np.eye(5))
    d = knn_distances(ref, ref, k=2, exclude_self=True)
    assert d.shape == (5, 2)
    assert np.all(d > 0.5), "the self-match (distance 0) must be dropped"


def test_exclude_self_on_a_single_entry_returns_no_columns():
    ref = _unit(np.eye(4)[:1])
    assert knn_distances(ref, ref, k=3, exclude_self=True).shape == (1, 0)


def test_novelty_from_distances_takes_the_k_smallest_across_blocks():
    a = np.array([[0.1, 0.9]], dtype=np.float32)
    b = np.array([[0.2, 0.8]], dtype=np.float32)
    # merged and sorted: 0.1 0.2 0.8 0.9 -> mean of the 3 smallest
    assert novelty_from_distances([a, b], k=3) == pytest.approx([(0.1 + 0.2 + 0.8) / 3])


def test_novelty_from_distances_with_no_reference_is_maximally_novel():
    out = novelty_from_distances([np.zeros((4, 0), np.float32)], k=10)
    assert np.array_equal(out, np.ones(4, np.float32))


def test_knn_novelty_is_the_mean_of_the_k_nearest_distances():
    ref = _unit(np.eye(4))
    q = _unit(np.array([[1.0, 0.0, 0.0, 0.0]]))
    # distances: 0, 1, 1, 1 -> k=3 mean = (0 + 1 + 1) / 3
    assert knn_novelty(q, ref, k=3) == pytest.approx([2.0 / 3.0], abs=1e-6)


def test_knn_novelty_on_an_empty_archive_is_one():
    out = knn_novelty(_unit(np.eye(3)), np.zeros((0, 3), np.float32), k=10)
    assert np.array_equal(out, np.ones(3, np.float32))


def test_rejects_ring_reports_its_contents():
    r = RejectsRing(capacity=4, dim=3)
    assert len(r) == 0
    r.add(_unit(np.eye(3)[:2]))
    assert len(r) == 2
    assert r.view().shape == (2, 3)
    assert r.view().dtype == np.float32


def test_rejects_ring_overwrites_oldest_first():
    r = RejectsRing(capacity=3, dim=2)
    for i in range(5):
        r.add(np.array([[float(i), 0.0]], dtype=np.float32))
    assert len(r) == 3
    assert sorted(r.view()[:, 0].tolist()) == [2.0, 3.0, 4.0]


def test_rejects_ring_clear():
    r = RejectsRing(capacity=4, dim=2)
    r.add(np.zeros((3, 2), np.float32))
    r.clear()
    assert len(r) == 0
    assert r.view().shape == (0, 2)


def test_sample_by_novelty_favours_high_novelty():
    rng = np.random.default_rng(0)
    nov = np.array([0.1, 0.9], dtype=np.float32)
    idx = sample_by_novelty(nov, 4000, rng, alpha=4.0)
    share = float((idx == 1).mean())
    # 0.9^4 / (0.9^4 + 0.1^4) = 0.99985
    assert share > 0.99


def test_sample_by_novelty_with_alpha_zero_is_uniform():
    rng = np.random.default_rng(1)
    nov = np.array([0.01, 0.5, 0.99], dtype=np.float32)
    idx = sample_by_novelty(nov, 6000, rng, alpha=0.0)
    counts = np.bincount(idx, minlength=3) / 6000.0
    assert np.allclose(counts, 1 / 3, atol=0.03)


def test_sample_by_novelty_falls_back_to_uniform_when_all_weights_vanish():
    rng = np.random.default_rng(2)
    idx = sample_by_novelty(np.zeros(3, np.float32), 3000, rng, alpha=4.0)
    counts = np.bincount(idx, minlength=3) / 3000.0
    assert np.allclose(counts, 1 / 3, atol=0.04)


def test_sample_by_novelty_ignores_negative_novelty():
    rng = np.random.default_rng(4)
    idx = sample_by_novelty(np.array([-1.0, 1.0], np.float32), 500, rng, alpha=4.0)
    assert np.all(idx == 1)


def test_sample_by_novelty_on_an_empty_archive_raises():
    with pytest.raises(ValueError, match="empty"):
        sample_by_novelty(np.zeros(0, np.float32), 3, np.random.default_rng(0))
