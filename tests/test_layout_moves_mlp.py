"""A layer stack does not fit "nudge one int", so mlp declares its own moves.

N is 1 by default: the point of the move is that the child is its parent plus
a little room, which is what makes the novelty it earns attributable. A large
jump is a restart wearing a growth move's name.
"""
from __future__ import annotations

import numpy as np

from services.brains import MAX_BRAIN_FLOATS, REGISTRY
from services.brains.layout_moves import (LayoutBounds, candidate_moves,
                                          propose_layout_move)
from services.brains.mlp import MAX_DEPTH, MAX_WIDTH, MIN_WIDTH

MLP = REGISTRY["mlp"]


def _lay(layers):
    return MLP.layout_from_settings({"layers": layers})


def _ops(layout, bounds=None):
    return sorted({m.operator for m in
                   candidate_moves(layout, bounds or LayoutBounds())})


def _children(layout, op, bounds=None):
    return sorted(m.child.signature() for m in
                  candidate_moves(layout, bounds or LayoutBounds())
                  if m.operator == op)


def test_a_single_layer_stack_offers_every_operator():
    lay = _lay([[16, 0]])
    assert _ops(lay) == ["activation", "add_layer", "grow", "shrink"]


def test_growing_adds_exactly_one_unit_to_one_layer():
    assert _children(_lay([[16, 0]]), "grow") == ["mlp-n17-a0"]
    # two layers, so two places to grow
    assert _children(_lay([[16, 0], [8, 0]]), "grow") == [
        "mlp-n16.9-a0.0", "mlp-n17.8-a0.0"]


def test_adding_a_layer_appends_one_at_the_default_width():
    """Appended, never inserted: an inserted layer renumbers every layer after
    it, so nothing downstream could be carried across."""
    assert _children(_lay([[16, 0]]), "add_layer") == ["mlp-n16.16-a0.0"]


def test_shrinking_and_dropping_are_separate_operators():
    lay = _lay([[16, 0], [8, 0]])
    assert _children(lay, "shrink") == ["mlp-n15.8-a0.0", "mlp-n16.7-a0.0"]
    assert _children(lay, "drop_layer") == ["mlp-n16-a0", "mlp-n8-a0"]


def test_the_last_layer_may_not_be_dropped():
    """A brain with no hidden layer is not a smaller brain, it is a different
    model."""
    assert _children(_lay([[16, 0]]), "drop_layer") == []


def test_changing_an_activation_keeps_every_width():
    got = _children(_lay([[16, 0]]), "activation")
    assert got == ["mlp-n16-a1", "mlp-n16-a2"]


def test_a_layer_at_the_floor_may_be_dropped_but_not_shrunk():
    """MIN_WIDTH is 1, and a width-0 layer is not a layer."""
    lay = _lay([[16, 0], [MIN_WIDTH, 0]])
    assert "mlp-n16-a0" in _children(lay, "drop_layer")
    # only the FIRST layer can still shrink
    assert _children(lay, "shrink") == ["mlp-n15.1-a0.0"]


def test_the_depth_bound_stops_add_layer():
    lay = _lay([[8, 0], [8, 0]])
    assert _children(lay, "add_layer", LayoutBounds(max_depth=2)) == []
    assert _children(lay, "add_layer", LayoutBounds(max_depth=3)) != []


def test_the_hard_depth_ceiling_holds_with_no_bound_set():
    lay = _lay([[MIN_WIDTH, 0]] * MAX_DEPTH)
    assert _children(lay, "add_layer") == []


def test_the_width_bound_stops_grow():
    lay = _lay([[8, 0]])
    assert _children(lay, "grow", LayoutBounds(max_width=8)) == []
    assert _children(lay, "grow", LayoutBounds(max_width=9)) == ["mlp-n9-a0"]


def test_a_move_that_the_clamp_would_change_is_rejected_not_silently_altered():
    """_shape_from_layers clamps to the float budget, so growing the second
    layer here asks for 1039 floats and comes back as mlp-n47.15 - a layout
    that also SHRANK the first layer. The round trip in candidate_moves is
    what turns that into no move at all rather than a move that did something
    else."""
    lay = _lay([[48, 0], [14, 0]])
    assert lay.signature() == "mlp-n48.14-a0.0", "the parent must not clamp"
    assert "mlp-n47.15-a0.0" not in _children(lay, "grow")
    assert "mlp-n48.15-a0.0" not in _children(lay, "grow")
    # growing the FIRST layer is over MAX_WIDTH, growing the second overflows,
    # so this stack cannot grow at all
    assert _children(lay, "grow") == []
    for m in candidate_moves(lay, LayoutBounds()):
        assert m.child.length <= MAX_BRAIN_FLOATS


def test_the_scales_survive_every_operator():
    lay = MLP.layout_from_settings({"layers": [[16, 0]], "w_scale": 4.0,
                                    "b_scale": 0.5})
    for m in candidate_moves(lay, LayoutBounds()):
        scales = dict(m.child.scales)
        assert (scales["w_scale"], scales["b_scale"]) == (4.0, 0.5)


def test_proposing_from_a_deep_stack_terminates():
    lay = _lay([[16, 0], [12, 1], [8, 2]])
    got = propose_layout_move(lay, LayoutBounds(), np.random.default_rng(1))
    assert got is not None and got.child.signature() != lay.signature()
