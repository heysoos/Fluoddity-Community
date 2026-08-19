"""Carrying a decoded brain into a neighbouring layout.

A grown child is BIT-IDENTICAL to its parent at birth. Every unit modality
stores an amplitude its evaluation multiplies by - `out += a * basis` - so a
new unit born with zero amplitude contributes nothing, and any novelty the
child earns is earned rather than an artefact of a random restart.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.brains import REGISTRY, default_layout
from services.brains.layout_moves import transfer_genome

FOURIER = default_layout()
GABOR = REGISTRY["gabor"].layout_from_settings({})
LENIA = REGISTRY["lenia"].layout_from_settings({})

# Where each modality's unit keeps the amplitude the evaluation multiplies by.
AMPLITUDE = {"fourier": (4, 8), "gabor": (8, 12), "lenia": (4, 8)}


def _parent(layout, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(REGISTRY[layout.modality].random(rng, layout),
                      dtype=np.float32).reshape(-1)


def _grow(layout, key, n):
    return REGISTRY[layout.modality].layout_from_settings({key: n})


def test_growing_keeps_every_parent_unit_verbatim():
    parent = _parent(FOURIER)
    child_layout = _grow(FOURIER, "centers", 11)
    got = transfer_genome(parent, FOURIER, child_layout,
                          np.random.default_rng(1))
    assert got.shape == (child_layout.length,)
    np.testing.assert_array_equal(got[:parent.size], parent)


def test_the_new_unit_is_born_silent():
    """Zero amplitude, so the child evaluates exactly as its parent does."""
    parent = _parent(FOURIER)
    child_layout = _grow(FOURIER, "centers", 11)
    got = transfer_genome(parent, FOURIER, child_layout,
                          np.random.default_rng(1))
    new = got[parent.size:]
    lo, hi = AMPLITUDE["fourier"]
    np.testing.assert_array_equal(new[lo:hi], np.zeros(hi - lo, np.float32))


def test_the_new_unit_is_not_inert_on_both_sides():
    """A unit zeroed everywhere takes generations of sigma to wake up. Its
    INCOMING half is drawn, so it has something to say the moment the search
    moves its amplitude."""
    parent = _parent(FOURIER)
    child_layout = _grow(FOURIER, "centers", 11)
    got = transfer_genome(parent, FOURIER, child_layout,
                          np.random.default_rng(1))
    new = got[parent.size:]
    lo, _hi = AMPLITUDE["fourier"]
    assert np.any(new[:lo] != 0.0), "the frequency half must be drawn"


@pytest.mark.parametrize("layout,key,grown", [
    (FOURIER, "centers", 11), (GABOR, "filters", 13), (LENIA, "bumps", 13)])
def test_every_unit_modality_transfers(layout, key, grown):
    parent = _parent(layout)
    child_layout = _grow(layout, key, grown)
    got = transfer_genome(parent, layout, child_layout,
                          np.random.default_rng(2))
    np.testing.assert_array_equal(got[:parent.size], parent)
    stride = REGISTRY[layout.modality].unit_floats(layout)
    new = got[parent.size:]
    assert new.size == stride
    lo, hi = AMPLITUDE[layout.modality]
    np.testing.assert_array_equal(new[lo:hi], np.zeros(hi - lo, np.float32))


def test_shrinking_keeps_the_units_that_remain():
    """Lossy by nature - a dropped unit's contribution goes with it - but what
    survives must survive unchanged."""
    parent = _parent(FOURIER)
    child_layout = _grow(FOURIER, "centers", 9)
    got = transfer_genome(parent, FOURIER, child_layout,
                          np.random.default_rng(3))
    assert got.shape == (child_layout.length,)
    np.testing.assert_array_equal(got, parent[:child_layout.length])


def test_a_transfer_is_float32():
    parent = _parent(FOURIER)
    child_layout = _grow(FOURIER, "centers", 11)
    got = transfer_genome(parent, FOURIER, child_layout,
                          np.random.default_rng(4))
    assert got.dtype == np.float32


@pytest.mark.parametrize("layout,key,grown", [
    (FOURIER, "centers", 11), (GABOR, "filters", 13), (LENIA, "bumps", 13)])
def test_the_transfer_re_encodes_without_clipping(layout, key, grown):
    """encode() is arctanh(p / scale) and clips at the rails. An AMPLITUDE is
    an offset-type float, so zero encodes to zero and a transferred genome
    makes a clean CMA-ES mean.

    Only the amplitude is zeroed, and that matters: every unit also carries a
    WIDTH (gabor's sigma, lenia's sigma) which does clip at zero, so a transfer
    that silenced the whole unit would start the search against a rail."""
    parent = _parent(layout)
    child_layout = _grow(layout, key, grown)
    got = transfer_genome(parent, layout, child_layout,
                          np.random.default_rng(5))
    _z, clipped = REGISTRY[layout.modality].encode(got, child_layout)
    assert clipped == 0


def test_a_cross_modality_pair_has_no_transfer():
    """A jump is a restart, and saying so is better than returning something
    that looks like a carried genome."""
    parent = _parent(FOURIER)
    with pytest.raises(ValueError, match="cross-modality"):
        transfer_genome(parent, FOURIER, GABOR, np.random.default_rng(6))


def test_an_unchanged_layout_returns_the_same_floats():
    parent = _parent(FOURIER)
    got = transfer_genome(parent, FOURIER, FOURIER, np.random.default_rng(7))
    np.testing.assert_array_equal(got, parent)
