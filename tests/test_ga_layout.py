"""The GA operator has to work for layouts that are not Fourier.

GAOptimizer reshaped z to (-1, 8) because services.genome.crossover expects
Fourier's (N, 8) centres. That fails outright for a length not divisible by 8 -
and is quietly WRONG for one that is, because it would cross over along rows
that are not units of anything.

The operator below was written correctly and then never connected: every test
here constructed GAOptimizer directly, while the only production route,
make_optimizer(), had no `layout` parameter at all. So `self._layout` was always
None and _breed always took its Fourier branch. Measured through make_optimizer
before the fix, tell() raised ValueError for gabor, lenia AND mlp - including
gabor at 168 and lenia at 120, both multiples of 8, because crossover's mask is
N_CENTERS=10 regardless.

The same defect as the settings CLAUDE.md flags, one level up: not a value that
is declared and never read, but a constructor argument that is implemented,
documented and tested, and never supplied.
"""
import numpy as np
import pytest

from services.brains import REGISTRY, default_layout
from services.optimizers import GAOptimizer, make_optimizer


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


def test_fourier_breeds_like_everything_else():
    """It used to get a per-centre branch. See
    test_crossover_treats_z_as_unstructured for why that was reasoning about
    the phenotype applied to z."""
    layout = default_layout()
    opt = GAOptimizer(layout.length, 8, 0.5, 0, None, layout=layout)
    opt.tell(opt.ask(8), np.arange(8, dtype=np.float32))
    assert opt.ask(8).shape == (8, layout.length)


def test_the_layout_does_not_change_how_z_is_bred():
    """The operator is layout-agnostic, so passing one must be indistinguishable
    from omitting it. If this ever starts failing, some structure has been
    reintroduced into z-space and it needs the same scrutiny as the last lot."""
    a = GAOptimizer(80, 8, 0.5, 0, None)
    b = GAOptimizer(80, 8, 0.5, 0, None, layout=default_layout())
    fit = np.arange(8, dtype=np.float32)
    a.tell(a.ask(8), fit)
    b.tell(b.ask(8), fit)
    assert np.array_equal(a.ask(8), b.ask(8))


# ---- the production route ----------------------------------------------
#
# Everything above builds a GAOptimizer by hand. The app reaches it only through
# make_optimizer(), which is where the layout went missing.


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_make_optimizer_hands_the_ga_its_layout(name):
    lay = REGISTRY[name].layout_from_settings({})
    opt = make_optimizer("GA", lay.length, 8, 0.5, 0, layout=lay)
    assert opt._layout is lay


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_the_ga_breeds_every_modality_through_make_optimizer(name):
    lay = REGISTRY[name].layout_from_settings({})
    opt = make_optimizer("GA", lay.length, 8, 0.5, 0, layout=lay)
    opt.tell(opt.ask(8), np.arange(8, dtype=np.float32))
    nxt = opt.ask(8)
    assert nxt.shape == (8, lay.length)
    assert np.all(np.isfinite(nxt))


@pytest.mark.parametrize("algo", ["CMA-ES", "Sep-CMA-ES", "Random Search"])
def test_the_layout_aware_signature_does_not_disturb_the_others(algo):
    """Only the GA has a structured operator; the rest search a plain vector and
    must accept - and ignore - the argument rather than raising on it."""
    lay = REGISTRY["gabor"].layout_from_settings({})
    opt = make_optimizer(algo, lay.length, 8, 0.5, 0, layout=lay)
    opt.tell(opt.ask(8), np.arange(8, dtype=np.float32))
    assert opt.ask(8).shape == (8, lay.length)


@pytest.mark.parametrize("centers", [4, 10, 20, 48])
def test_the_ga_breeds_any_fourier_width(centers):
    """services.genome.crossover masks with a hardcoded N_CENTERS=10, so the
    per-centre branch worked at exactly the default width and raised at every
    other. Measured: 4, 20 and 48 centres all ValueError."""
    lay = REGISTRY["fourier"].layout_from_settings({"centers": centers})
    opt = make_optimizer("GA", lay.length, 8, 0.5, 0, layout=lay)
    opt.tell(opt.ask(8), np.arange(8, dtype=np.float32))
    assert opt.ask(8).shape == (8, lay.length)


def test_crossover_treats_z_as_unstructured():
    """z is NOT the phenotype, and the per-centre branch assumed it was.

    FourierModality.decode reads z as [all N frequencies, then all N
    amplitudes]. Verified by driving one z index at a time: z[0:4] is centre 0's
    frequency, z[4:8] is centre 1's FREQUENCY, and z[40:44] is centre 0's
    amplitude. The (-1, 8) reshape called z[4:8] an amplitude and z[40:44] a
    frequency, so half the frequencies got the amplitude operator and vice
    versa - and a crossed "centre" was really two centres' frequency vectors.

    No layout can rescue that, because it is not a property of the modality: the
    squash decides the z ordering. Per-gene is the honest operator here, and it
    is the one every modality now gets.
    """
    lay = REGISTRY["fourier"].layout_from_settings({})
    opt = make_optimizer("GA", lay.length, 8, 0.5, 0, layout=lay)
    z = opt.ask(8)
    fit = np.zeros(8, dtype=np.float32)
    fit[: GAOptimizer.ELITES] = np.arange(GAOptimizer.ELITES, 0, -1)
    z[: GAOptimizer.ELITES // 2] = -5.0
    z[GAOptimizer.ELITES // 2: GAOptimizer.ELITES] = 5.0
    opt.tell(z, fit)
    kids = opt.ask(8)[GAOptimizer.ELITES:]
    blocks = kids.reshape(len(kids), -1, 8)
    split = ((blocks > 0).any(axis=2) & (blocks < 0).any(axis=2))
    assert split.any(), "every 8-float block came whole - still crossing rows"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_a_ga_search_runs_end_to_end_for_every_modality(name):
    """The wiring, not the operator: a driver built on a non-Fourier spec has to
    reach tell() without raising.

    The grid must exceed ELITES. At 4 tiles the elites alone fill the next
    population, _breed() is never called, and this passes without testing
    anything - which is how it read on the first run.
    """
    from services.genome_spec import spec_for
    from services.prompt_driver import PromptDriver
    from services.tournament_service import TournamentService

    lay = REGISTRY[name].layout_from_settings({})
    n = 4 * GAOptimizer.ELITES
    ts = TournamentService(grid=4)
    ts.init_population()
    d = PromptDriver(ts, scorer=None, spec=spec_for(lay))
    d.algorithm = "GA"
    z = d.ask(n)
    assert z.shape == (n, lay.length)
    d.tell(z, [])
    assert d.ask(n).shape == (n, lay.length)
