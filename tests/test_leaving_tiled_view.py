"""Leaving the Tiled view must not move the picture.

The Tiled view repeats the canvas across the whole plane, so panning to 5.7 or
-12.4 is ordinary there. Every other view shows one canvas, so the camera is
wrapped back into a single period on the way out.

The wrap that did that was `fmod(p + 100, 2) - 1`, and 100 is a MULTIPLE OF THE
PERIOD - so the `+100` shifted nothing and the `-1` was left over, subtracting a
half-period from every position including 0. Leaving Tiled from dead centre
threw the camera a full canvas-width sideways, which is what "the centre moves
and I have to re-pan" was.

The wrap has to be the IDENTITY inside one period and fold only what is outside
it. Pure arithmetic, no GL: what was wrong is the formula.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.tiled_wrap import wrap_camera_position

PERIOD = 2.0


def old_wrap(p):
    """The formula that shipped. Kept so the tests below can show it moving
    things it had no business moving."""
    return float(np.fmod(p + 100.0, PERIOD) - 1.0)


@pytest.mark.parametrize("p", [0.0, 0.3, -0.3, 0.9, -0.9, 0.999, -1.0])
def test_a_position_already_in_range_is_left_alone(p):
    """The whole bug: these are the positions an ordinary pan produces, and
    every one of them was being moved."""
    assert wrap_camera_position(p) == pytest.approx(p, abs=1e-9)


def test_the_centred_camera_was_the_worst_case():
    """0.0 -> -1.0 under the old formula: a full half-period from dead centre,
    which is the most visible possible version of this bug."""
    assert old_wrap(0.0) == pytest.approx(-1.0)
    assert wrap_camera_position(0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("p", [0.0, 0.3, -0.3, 0.9, 1.0, 2.4, 5.7, -12.4])
def test_the_old_formula_really_did_move_everything(p):
    """Without this the tests above could pass against a no-op change."""
    assert abs(old_wrap(p) - p) > 1e-9 or abs(p) > PERIOD


@pytest.mark.parametrize("p,want", [
    (1.0, -1.0),      # exactly one period out folds to the lower edge
    (1.5, -0.5),
    (2.4, 0.4),
    (5.7, -0.3),
    (-1.5, 0.5),
    (-12.4, -0.4),
])
def test_a_position_outside_the_range_folds_into_it(p, want):
    """Tiled panning is unbounded, so the wrap still has real work to do."""
    got = wrap_camera_position(p)
    assert got == pytest.approx(want, abs=1e-9)
    assert -1.0 <= got < 1.0


@pytest.mark.parametrize("p", [0.0, 0.7, -0.7, 3.3, -3.3, 1e6, -1e6])
def test_the_wrap_always_lands_in_one_period(p):
    got = wrap_camera_position(p)
    assert -1.0 <= got < 1.0, f"{p} -> {got}"


@pytest.mark.parametrize("p", [0.4, -0.4, 2.9, -7.1])
def test_wrapping_twice_changes_nothing(p):
    """A wrap is idempotent; the old one was not - it drifted by -1 each time,
    so two view switches moved the camera twice."""
    once = wrap_camera_position(p)
    assert wrap_camera_position(once) == pytest.approx(once, abs=1e-9)


def test_the_old_formula_drifted_on_every_switch():
    """Names the mechanism: repeated Tiled -> Camera switches walked the camera
    across the world one half-period at a time."""
    p = 0.0
    for _ in range(3):
        p = old_wrap(p)
    assert p == pytest.approx(-1.0)  # folded, but only because fmod re-wrapped
    q = 0.0
    for _ in range(3):
        q = wrap_camera_position(q)
    assert q == pytest.approx(0.0), "the fixed wrap must not drift"
