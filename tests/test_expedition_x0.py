"""An expedition can be told WHERE to start, not only which row to start at.

A genome carried across a layout move has no archive row - it is a brain of a
layout the archive may hold nothing of yet - so _x0_index is None and the
optimizer's mean comes in directly.
"""
from __future__ import annotations

import numpy as np

from tests.test_imgep_driver import DIM, make, snaps


def _goal():
    g = np.zeros(DIM, dtype=np.float32)
    g[0] = 1.0
    return g


def test_an_explicit_x0_becomes_the_optimizers_mean():
    d, _arc, _ts = make()
    x0 = np.full(d.spec.dim, 0.25, dtype=np.float32)
    assert d.start_expedition_at(_goal(), "latent", "", x0) is True
    np.testing.assert_allclose(d.optimizer._x0, x0, atol=1e-6)


def test_an_explicit_x0_needs_no_archive_row_at_all():
    """The whole point: an empty archive cannot supply a seed, and a
    transferred genome does not need one."""
    d, arc, _ts = make()
    assert len(arc) == 0
    x0 = np.zeros(d.spec.dim, dtype=np.float32)
    assert d.start_expedition_at(_goal(), "latent", "", x0) is True
    assert d.regime == "expedition"
    assert d._x0_index is None


def test_an_x0_of_the_wrong_width_is_refused():
    """Refused rather than sliced - the same discipline GenomeSpec._check_width
    applies, and for the same reason."""
    d, _arc, _ts = make()
    bad = np.zeros(d.spec.dim + 1, dtype=np.float32)
    assert d.start_expedition_at(_goal(), "latent", "", bad) is False
    assert d.regime != "expedition"


def test_start_expedition_with_still_seeds_from_its_row():
    """The wrapper must be unchanged in behaviour: every existing caller goes
    through it."""
    d, arc, _ts = make(grid=2, seed_n=1)
    d.tell(d.ask(4), snaps(4, [[10, 20, 30, 40], [11, 21, 31, 41]]))
    assert len(arc) > 0
    assert d.start_expedition_with(arc.embeddings[0].copy(), "chase", "") is True
    assert d._x0_index is not None
    np.testing.assert_allclose(
        d.optimizer._x0, d._parent_z(d._x0_index), atol=1e-5)


def test_a_goal_naming_no_row_and_no_embedding_still_declines():
    """A novelty goal with an out-of-range seed has nothing to point at and
    nowhere to start."""
    d, _arc, _ts = make()
    assert d.start_expedition_with(None, "novelty", "", seed_index=99) is False
