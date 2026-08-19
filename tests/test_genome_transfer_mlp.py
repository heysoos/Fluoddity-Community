"""Repacking a layer stack into a neighbouring one.

A hidden unit is three separate regions of the buffer - its input weights, its
bias, and its COLUMN of the next layer's matrix - and W_out is stored
OUTPUT-MAJOR, so the new column is strided writes rather than a contiguous
append. Offsets come from layer_spans, the one definition mlp.glsl is checked
against.
"""
from __future__ import annotations

import numpy as np

from services.brains import REGISTRY
from services.brains.layout_moves import transfer_genome
from services.brains.mlp import layer_spans

MLP = REGISTRY["mlp"]


def _lay(layers):
    return MLP.layout_from_settings({"layers": layers})


def _parent(layout, seed=0):
    return np.asarray(MLP.random(np.random.default_rng(seed), layout),
                      dtype=np.float32).reshape(-1)


def _unpack(p, shape):
    hidden, out_w, out_b, n = layer_spans(shape)
    assert p.size == n
    layers = []
    for w_off, b_off, fan_in, w in hidden:
        layers.append((p[w_off:b_off].reshape(w, fan_in),
                       p[b_off:b_off + w].copy()))
    fan = hidden[-1][3] if hidden else 4
    return layers, p[out_w:out_b].reshape(4, fan), p[out_b:].copy()


def test_growing_the_only_layer_keeps_every_parent_weight():
    pa, ch = _lay([[16, 0]]), _lay([[17, 0]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(1))
    assert got.size == ch.length

    lp, Wop, bop = _unpack(p, pa.shape)
    lc, Woc, boc = _unpack(got, ch.shape)
    np.testing.assert_array_equal(lc[0][0][:16], lp[0][0])
    np.testing.assert_array_equal(lc[0][1][:16], lp[0][1])
    np.testing.assert_array_equal(Woc[:, :16], Wop)
    np.testing.assert_array_equal(boc, bop)


def test_the_new_units_outgoing_column_is_zero():
    """W_out is output-major, so the column is one strided entry per output.
    Zero there is what makes the child identical to its parent at birth."""
    pa, ch = _lay([[16, 0]]), _lay([[17, 0]])
    got = transfer_genome(_parent(pa), pa, ch, np.random.default_rng(1))
    _lc, Woc, _boc = _unpack(got, ch.shape)
    np.testing.assert_array_equal(Woc[:, 16], np.zeros(4, np.float32))


def test_the_new_units_incoming_weights_are_drawn():
    pa, ch = _lay([[16, 0]]), _lay([[17, 0]])
    got = transfer_genome(_parent(pa), pa, ch, np.random.default_rng(1))
    lc, _Woc, _boc = _unpack(got, ch.shape)
    assert np.any(lc[0][0][16] != 0.0)


def test_growing_a_middle_layer_widens_the_NEXT_layers_fan_in():
    """Layer l gaining a unit adds a COLUMN to W_{l+1}, which is a hidden
    matrix here rather than W_out."""
    pa, ch = _lay([[16, 0], [8, 0]]), _lay([[17, 0], [8, 0]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(2))
    lp, Wop, _ = _unpack(p, pa.shape)
    lc, Woc, _ = _unpack(got, ch.shape)
    np.testing.assert_array_equal(lc[1][0][:, :16], lp[1][0])
    np.testing.assert_array_equal(lc[1][0][:, 16], np.zeros(8, np.float32))
    np.testing.assert_array_equal(Woc, Wop)


def test_adding_a_layer_leaves_the_parents_stack_untouched():
    pa, ch = _lay([[16, 0]]), _lay([[16, 0], [16, 0]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(3))
    lp, _Wop, _ = _unpack(p, pa.shape)
    lc, _Woc, _ = _unpack(got, ch.shape)
    np.testing.assert_array_equal(lc[0][0], lp[0][0])
    np.testing.assert_array_equal(lc[0][1], lp[0][1])


def test_shrinking_keeps_the_units_that_remain():
    pa, ch = _lay([[16, 0]]), _lay([[15, 0]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(4))
    lp, Wop, _ = _unpack(p, pa.shape)
    lc, Woc, _ = _unpack(got, ch.shape)
    np.testing.assert_array_equal(lc[0][0], lp[0][0][:15])
    np.testing.assert_array_equal(Woc, Wop[:, :15])


def test_dropping_a_layer_keeps_the_layers_that_remain():
    pa, ch = _lay([[16, 0], [8, 1]]), _lay([[16, 0]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(5))
    lp, _Wop, _ = _unpack(p, pa.shape)
    lc, _Woc, _ = _unpack(got, ch.shape)
    np.testing.assert_array_equal(lc[0][0], lp[0][0])


def test_an_activation_change_moves_no_floats():
    """Same widths, same offsets - only what the shader does between them."""
    pa, ch = _lay([[16, 0]]), _lay([[16, 2]])
    p = _parent(pa)
    got = transfer_genome(p, pa, ch, np.random.default_rng(6))
    np.testing.assert_array_equal(got, p)


def test_the_transfer_re_encodes_without_clipping():
    pa, ch = _lay([[16, 0]]), _lay([[17, 0]])
    got = transfer_genome(_parent(pa), pa, ch, np.random.default_rng(7))
    _z, clipped = MLP.encode(got, ch)
    assert clipped == 0


def test_every_operator_produces_the_right_width():
    from services.brains.layout_moves import LayoutBounds, candidate_moves

    pa = _lay([[16, 0], [8, 1]])
    p = _parent(pa)
    for mv in candidate_moves(pa, LayoutBounds()):
        got = transfer_genome(p, pa, mv.child, np.random.default_rng(8))
        assert got.size == mv.child.length, mv.operator
