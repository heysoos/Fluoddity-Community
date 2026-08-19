"""A grown brain must BE its parent until the search moves it.

The whole point of transferring rather than redrawing: if a child's novelty
came from a random restart, a layout move would look productive whatever shape
it proposed. This is the only test that can see it, because the phenotype is
what the SHADER computes and a NumPy check compares Python against itself.

Needs a real GL context, so it skips where there is none.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.brains import REGISTRY, default_layout
from services.brains.layout_moves import (LayoutBounds, candidate_moves,
                                          transfer_genome)
from tests.test_brain_modalities_gpu import evaluate, gl, inputs  # noqa: F401

# The ONE operator that can be phenotype-preserving. A dropped unit takes its
# contribution with it, and an activation change is a different function by
# definition - but `add_layer` is the interesting exclusion, and it is a fact
# about the architecture rather than a limit of the transfer: appending a layer
# puts a new NONLINEARITY between the old last hidden layer and W_out, and none
# of tanh, sin or gelu has an identity region to pass the signal through
# unchanged. See test_adding_a_layer_is_a_discontinuous_move below.
GROWTH = ("grow",)

CASES = [
    default_layout(),
    REGISTRY["gabor"].layout_from_settings({}),
    REGISTRY["lenia"].layout_from_settings({}),
    REGISTRY["mlp"].layout_from_settings({"layers": [[16, 0]]}),
    REGISTRY["mlp"].layout_from_settings({"layers": [[12, 0], [8, 1]]}),
]


@pytest.mark.gpu
@pytest.mark.parametrize("parent", CASES,
                         ids=[c.signature() for c in CASES])
def test_a_growth_move_does_not_change_what_the_brain_computes(gl, parent):
    xs = inputs(3)
    p = np.asarray(REGISTRY[parent.modality].random(
        np.random.default_rng(11), parent), dtype=np.float32).reshape(-1)
    before = evaluate(gl, p, parent, xs)

    seen = 0
    for mv in candidate_moves(parent, LayoutBounds()):
        if mv.operator not in GROWTH:
            continue
        seen += 1
        child = transfer_genome(p, parent, mv.child, np.random.default_rng(12))
        after = evaluate(gl, child, mv.child, xs)
        np.testing.assert_array_equal(
            after, before,
            err_msg=f"{mv.operator}: {parent.signature()} -> "
                    f"{mv.child.signature()} changed the phenotype")
    assert seen, f"no growth move from {parent.signature()}"


@pytest.mark.gpu
def test_adding_a_layer_is_a_discontinuous_move(gl):
    """Stated as a PROPERTY, because it looks like a bug and is not.

    A deeper stack cannot be born identical: the new layer's activation sits
    between the old last hidden layer and W_out, and tanh, sin and gelu are all
    nonlinear at every point - there is no identity region to pass through.
    Net2DeeperNet does this with ReLU, where ReLU(x) = x for x > 0.

    So the parent's EARLIER layers are carried and the rest is drawn: better
    than a full restart, but the search should treat a depth change as the
    jump it is.
    """
    from services.brains.mlp import layer_spans

    parent = REGISTRY["mlp"].layout_from_settings({"layers": [[16, 0]]})
    child_layout = REGISTRY["mlp"].layout_from_settings(
        {"layers": [[16, 0], [16, 0]]})
    xs = inputs(5)
    p = np.asarray(REGISTRY["mlp"].random(
        np.random.default_rng(15), parent), dtype=np.float32).reshape(-1)
    before = evaluate(gl, p, parent, xs)
    child = transfer_genome(p, parent, child_layout, np.random.default_rng(16))
    after = evaluate(gl, child, child_layout, xs)
    assert not np.array_equal(after, before)

    # ...but the first layer really did come across, which is the half that is
    # not a restart.
    p_hidden, _pow, _pob, _pn = layer_spans(parent.shape)
    c_hidden, _cow, _cob, _cn = layer_spans(child_layout.shape)
    pw, pb, _pf, _pw = p_hidden[0]
    cw, cb, _cf, _cw = c_hidden[0]
    np.testing.assert_array_equal(child[cw:cb], p[pw:pb])


@pytest.mark.gpu
def test_a_shrink_move_is_allowed_to_differ(gl):
    """Stated so the equality above is read as a property of GROWTH rather
    than an accident of these layouts."""
    parent = default_layout()
    xs = inputs(4)
    p = np.asarray(REGISTRY["fourier"].random(
        np.random.default_rng(13), parent), dtype=np.float32).reshape(-1)
    before = evaluate(gl, p, parent, xs)
    child_layout = REGISTRY["fourier"].layout_from_settings({"centers": 9})
    child = transfer_genome(p, parent, child_layout, np.random.default_rng(14))
    after = evaluate(gl, child, child_layout, xs)
    assert not np.array_equal(after, before)
