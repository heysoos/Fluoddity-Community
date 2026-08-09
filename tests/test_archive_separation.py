"""The archive stores a COVERING, not a log.

A converging expedition proposes the same pattern 64 times a generation for 50
generations. Measured on the real archives, one goal had already contributed
25% of debug07 (3200 of 12672) and 30% of debug05. The separation rule - the
unstructured-archive rule from quality-diversity - refuses anything within `l`
of something already stored, and the driver forces the best tile of every
generation through it so a run always leaves a trail.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.archive import DEFAULT_MIN_SEPARATION, Archive, Candidate
from services.novelty import nearest_distance


def arc(**kw):
    kw.setdefault("dim", 4)
    kw.setdefault("capacity", 1000)
    kw.setdefault("liveness_min", 0.0)
    return Archive(store=None, **kw)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def cand(vec, liveness=1.0):
    e = np.zeros(4, dtype=np.float32)
    e[: len(vec)] = unit(vec)
    return Candidate(brain=np.zeros((10, 8), np.float32),
                     physics=np.zeros(8, np.float32), embedding=e,
                     liveness=liveness, spec="t", viable=True)


# ---- the primitive ------------------------------------------------------

def test_nearest_distance_is_the_closest_not_the_mean():
    """The whole point of a separate primitive. Ten neighbours at 0.5 and one
    at 0.001 give a healthy-looking kNN novelty for something the archive
    already holds; the 1-NN distance says 0.001."""
    ref = np.stack([unit([1, 0, 0, 0])] + [unit([0, 1, 0, 0])] * 9)
    q = unit([1.0, 0.001, 0, 0])[None, :]
    assert float(nearest_distance(q, ref)[0]) < 1e-4


def test_nearest_distance_of_an_empty_reference_is_infinite():
    d = nearest_distance(np.zeros((3, 4), np.float32), np.zeros((0, 4), np.float32))
    assert np.isinf(d).all(), "nothing is nearby when there is nothing"


# ---- the rule -----------------------------------------------------------

def test_a_duplicate_is_refused():
    a = arc(min_separation=0.05)
    assert a.consider(cand([1, 0, 0, 0]), novelty=1.0) is not None
    assert a.consider(cand([1, 0, 0, 0]), novelty=1.0) is None
    assert len(a) == 1
    assert a.stats()["n_rejected_close"] == 1


def test_something_genuinely_different_still_gets_in():
    a = arc(min_separation=0.05)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    assert a.consider(cand([0, 1, 0, 0]), novelty=1.0) is not None
    assert len(a) == 2


def test_the_threshold_is_where_it_says_it_is():
    """Just inside is refused, just outside is admitted. Cosine distance, so
    a rotation of theta gives 1 - cos(theta)."""
    l = 0.05
    a = arc(min_separation=l)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    for frac, want_in in ((0.9, False), (1.1, True)):
        th = float(np.arccos(1.0 - l * frac))
        b = arc(min_separation=l)
        b.consider(cand([1, 0, 0, 0]), novelty=1.0)
        got = b.consider(cand([np.cos(th), np.sin(th), 0, 0]), novelty=1.0)
        assert (got is not None) is want_in, f"at {frac}x the threshold"


def test_zero_disables_the_rule_entirely():
    a = arc(min_separation=0.0)
    for _ in range(5):
        a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    assert len(a) == 5


def test_force_bypasses_separation():
    a = arc(min_separation=0.5)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    assert a.consider(cand([1, 0, 0, 0]), novelty=1.0, force=True) is not None


def test_force_does_NOT_bypass_liveness():
    """Keeping one entry per generation is worth doing; keeping a frozen
    canvas is not, and a forced dud would be admitted every generation."""
    a = arc(min_separation=0.5, liveness_min=0.5)
    assert a.consider(cand([1, 0, 0, 0], liveness=0.0), novelty=1.0,
                      force=True) is None


def test_a_pin_is_never_refused():
    a = arc(min_separation=0.9)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    assert a.consider(cand([1, 0, 0, 0]), novelty=1.0, pinned=True) is not None


def test_a_close_rejection_does_not_poison_the_rejects_ring():
    """The ring is 2048 entries of memory about regions the search was REFUSED.
    A separation rejection means the region is in the archive, which novelty
    measures against anyway - feeding those in would flush the dead regions out
    of the ring within a couple of generations at grid 8."""
    a = arc(min_separation=0.5)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    assert len(a.rejects) == 0
    a.liveness_min = 0.5
    a.consider(cand([0, 1, 0, 0], liveness=0.0), novelty=1.0)
    assert len(a.rejects) == 1, "a dead tile still belongs in the ring"


def test_rejection_reasons_are_counted_apart():
    """They mean opposite things: 'dead' rising is a fault, 'too close' rising
    is the rule working."""
    a = arc(min_separation=0.5, liveness_min=0.5)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    a.consider(cand([0, 1, 0, 0], liveness=0.0), novelty=1.0)
    st = a.stats()
    assert (st["n_rejected_close"], st["n_rejected_dead"]) == (1, 1)


def test_a_precomputed_separation_is_used_as_given():
    """The driver does one matmul for the whole batch instead of n
    matrix-vector products; passing the answer in has to mean the same thing."""
    a = arc(min_separation=0.05)
    a.consider(cand([1, 0, 0, 0]), novelty=1.0)
    assert a.consider(cand([0, 1, 0, 0]), novelty=1.0, separation=0.01) is None
    assert a.consider(cand([0, 0, 1, 0]), novelty=1.0, separation=0.9) is not None


def test_the_default_is_the_measured_one():
    assert Archive(store=None).min_separation == DEFAULT_MIN_SEPARATION
    assert DEFAULT_MIN_SEPARATION == pytest.approx(0.02)


def test_mean_novelty_is_reported_and_empty_is_zero():
    a = arc(min_separation=0.0)
    assert a.mean_novelty() == 0.0
    a.consider(cand([1, 0, 0, 0]), novelty=0.4)
    a.consider(cand([0, 1, 0, 0]), novelty=0.6)
    assert a.mean_novelty() == pytest.approx(0.5)
