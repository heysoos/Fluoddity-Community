"""Proposing a neighbouring brain shape, under bounds.

A move is PROPOSED, REBUILT and COMPARED - the round-trip discipline
layout_from_signature already uses - because _shape_from_layers CLAMPS rather
than raises, so a proposal that hits a limit comes back as a DIFFERENT layout
instead of an error.
"""
from __future__ import annotations

import numpy as np

from services.brains import REGISTRY, BrainLayout, default_layout
from services.brains.layout_moves import (LayoutBounds, candidate_moves,
                                          propose_layout_move)

FOURIER = default_layout()                      # fourier-n10
GABOR = REGISTRY["gabor"].layout_from_settings({})
LENIA = REGISTRY["lenia"].layout_from_settings({})


def _rng():
    return np.random.default_rng(0)


def _ops(layout, bounds=None):
    return sorted(m.operator for m in
                  candidate_moves(layout, bounds or LayoutBounds()))


def _by_op(layout, op, bounds=None):
    for m in candidate_moves(layout, bounds or LayoutBounds()):
        if m.operator == op:
            return m
    raise AssertionError(f"no {op!r} move from {layout.signature()}")


def test_one_integer_of_structure_moves_by_one():
    """The three unit modalities need no hook at all: their structure IS one
    integer, so +-1 on it is the whole operator set."""
    assert _ops(FOURIER) == ["grow", "shrink"]
    grow = _by_op(FOURIER, "grow")
    assert grow.child.signature() == "fourier-n11"
    assert grow.child.length == 8 * 11
    shrink = _by_op(FOURIER, "shrink")
    assert shrink.child.signature() == "fourier-n9"


def test_every_unit_modality_moves():
    for lay, grown in ((GABOR, "gabor-n13"), (LENIA, "lenia-n13")):
        assert _by_op(lay, "grow").child.signature() == grown


def test_a_move_keeps_the_decode_scales():
    """Scales are not structure. A move that reset them would silently change
    what every transferred float MEANS."""
    tuned = BrainLayout("fourier", (10,), 80,
                        scales=(("freq_scale", 3.5), ("low_freq_bias", 0.25)))
    child = _by_op(tuned, "grow").child
    assert dict(child.scales) == {"freq_scale": 3.5, "low_freq_bias": 0.25}


def test_the_modality_own_floor_and_ceiling_are_respected():
    """Fourier declares centers 4..48, so neither end proposes past it."""
    at_floor = REGISTRY["fourier"].layout_from_settings({"centers": 4})
    assert _ops(at_floor) == ["grow"]
    at_ceiling = REGISTRY["fourier"].layout_from_settings({"centers": 48})
    assert _ops(at_ceiling) == ["shrink"]


def test_the_users_width_bound_narrows_it_further():
    bounds = LayoutBounds(max_width=11)
    assert _ops(FOURIER, bounds) == ["grow", "shrink"]
    at_bound = REGISTRY["fourier"].layout_from_settings({"centers": 11})
    assert _ops(at_bound, bounds) == ["shrink"]


def test_the_float_budget_is_checked_before_a_layout_is_built():
    """BrainLayout.__post_init__ RAISES past MAX_BRAIN_FLOATS, so a proposal
    that would cross it must never reach the constructor."""
    bounds = LayoutBounds(max_floats=8 * 10)      # exactly the parent's size
    assert _ops(FOURIER, bounds) == ["shrink"]


def test_a_banned_pair_is_skipped():
    banned = {("fourier-n10", "fourier-n11")}
    got = {propose_layout_move(FOURIER, LayoutBounds(), _rng(), banned)
           .child.signature() for _ in range(20)}
    assert got == {"fourier-n9"}


def test_when_every_move_is_banned_it_returns_nothing():
    """The caller runs an ordinary expedition instead. It must not fall back
    to a worse move and it must not raise."""
    banned = {("fourier-n10", "fourier-n11"), ("fourier-n10", "fourier-n9")}
    assert propose_layout_move(FOURIER, LayoutBounds(), _rng(), banned) is None


def test_a_move_never_returns_the_parent():
    for m in candidate_moves(FOURIER, LayoutBounds()):
        assert m.child.signature() != m.parent.signature()
        assert m.parent is FOURIER


def test_proposing_is_seeded_and_reproducible():
    a = propose_layout_move(FOURIER, LayoutBounds(), np.random.default_rng(4))
    b = propose_layout_move(FOURIER, LayoutBounds(), np.random.default_rng(4))
    assert a == b


def test_a_modality_with_no_structural_int_proposes_nothing():
    """Not a failure - it is how a modality opts out until it declares a hook."""
    class _Flat:
        name = "flat"

        def settings_schema(self):
            from services.brains import Setting
            return [Setting("gain", "Gain", "float", 0.0, 1.0, 0.5)]

    lay = BrainLayout("flat", (), 8)
    assert candidate_moves(lay, LayoutBounds(), modality=_Flat()) == []
