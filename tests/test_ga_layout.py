"""The GA operator has to work for layouts that are not Fourier.

GAOptimizer reshaped z to (-1, 8) because services.genome.crossover expects
Fourier's (N, 8) centres. That fails outright for a length not divisible by 8 -
and is quietly WRONG for one that is, because it would cross over along rows
that are not units of anything.
"""
import numpy as np

from services.brains import REGISTRY, default_layout
from services.optimizers import GAOptimizer


def test_a_length_that_is_not_a_multiple_of_eight_breeds():
    layout = REGISTRY["mlp"].layout_from_settings({"hidden": 16})
    assert layout.length % 8 != 0, "pick a width that actually exercises this"

    opt = GAOptimizer(layout.length, 8, 0.5, 0, None, layout=layout)
    opt.tell(opt.ask(8), np.arange(8, dtype=np.float32))
    nxt = opt.ask(8)
    assert nxt.shape == (8, layout.length)
    assert np.all(np.isfinite(nxt))


def test_every_modality_can_be_bred():
    for m in REGISTRY.values():
        layout = m.layout_from_settings({})
        opt = GAOptimizer(layout.length, 6, 0.5, 1, None, layout=layout)
        opt.tell(opt.ask(6), np.arange(6, dtype=np.float32))
        got = opt.ask(6)
        assert got.shape == (6, layout.length), m.name
        assert np.all(np.isfinite(got)), m.name


def test_a_generic_child_takes_each_gene_from_one_parent_or_the_other():
    """Uniform per-gene crossover is the honest generic operator: without a
    meaningful row structure, blending genes would invent values neither parent
    had. Only the mutation may move a gene off a parent's value."""
    layout = REGISTRY["lenia"].layout_from_settings({})
    opt = GAOptimizer(layout.length, 8, 0.5, 3, None, layout=layout)
    z = opt.ask(8)
    # EVERY elite must be pinned, not just two: breeding draws both parents
    # from the top ELITES, so an unpinned one contributes genes near zero and
    # looks exactly like blending.
    fit = np.zeros(8, dtype=np.float32)
    fit[: GAOptimizer.ELITES] = np.arange(GAOptimizer.ELITES, 0, -1)
    z[: GAOptimizer.ELITES // 2] = -5.0
    z[GAOptimizer.ELITES // 2: GAOptimizer.ELITES] = 5.0
    opt.tell(z, fit)
    kids = opt.ask(8)[GAOptimizer.ELITES:]
    off = np.minimum(np.abs(kids - 5.0), np.abs(kids + 5.0))
    assert off.max() < 4.0 * GAOptimizer.MUT + 1e-3, (
        "a child gene came from neither parent - the operator is blending"
    )


def test_fourier_still_breeds_per_centre():
    """A Fourier centre IS a unit, so per-centre crossover is meaningful there
    and must survive the generalisation."""
    layout = default_layout()
    opt = GAOptimizer(layout.length, 8, 0.5, 0, None, layout=layout)
    opt.tell(opt.ask(8), np.arange(8, dtype=np.float32))
    assert opt.ask(8).shape == (8, layout.length)


def test_omitting_the_layout_reproduces_the_old_behaviour():
    """Every existing call site omits `layout`; those must be untouched."""
    a = GAOptimizer(80, 8, 0.5, 0, None)
    b = GAOptimizer(80, 8, 0.5, 0, None, layout=default_layout())
    fit = np.arange(8, dtype=np.float32)
    a.tell(a.ask(8), fit)
    b.tell(b.ask(8), fit)
    assert np.array_equal(a.ask(8), b.ask(8)), (
        "passing the Fourier layout explicitly must be the same as omitting it"
    )
