"""A modality round trip must land back where it started.

Switching fourier -> gabor -> fourier gave a different set of tiles every time,
because TournamentService took an UNSEEDED np.random.default_rng() and
set_layout rerolls the whole population. The seed is the app's control over
randomness - it already decides the no-rule cohort brains through
generated_brains - so the tiles have to answer to it too.

Reset is deliberately NOT seeded to a fixed draw: it is the "give me other
options" button, and a reset that returned the same sixteen tiles forever would
be a broken control. It stays reproducible in the weaker sense that matters -
the whole SEQUENCE of resets replays for a given seed.
"""
import numpy as np
import pytest

from services.brains import REGISTRY, brain_rng, generated_brains
from services.tournament_service import TournamentService

MODALITIES = sorted(REGISTRY)


def layout(name, **settings):
    return REGISTRY[name].layout_from_settings(settings)


def widths(ts):
    return [np.asarray(g).copy() for g in ts.population]


def same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_the_seed_mapping_has_one_home():
    """brain_rng is what generated_brains uses; a second copy of the formula is
    how the tiles and the cohort brains would drift apart."""
    a = brain_rng(0.25).standard_normal(8)
    b = brain_rng(0.25).standard_normal(8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, brain_rng(0.26).standard_normal(8))


@pytest.mark.parametrize("name", ["gabor", "lenia", "mlp"])
def test_a_modality_round_trip_returns_the_same_tiles(name):
    ts = TournamentService(grid=4, layout=layout("fourier"), seed=0.5)
    ts.init_population()
    before = widths(ts)
    ts.set_layout(layout(name))
    ts.set_layout(layout("fourier"))
    assert same(before, widths(ts))


def test_a_round_trip_through_every_modality_returns_the_same_tiles():
    ts = TournamentService(grid=4, layout=layout("fourier"), seed=0.5)
    ts.init_population()
    before = widths(ts)
    for name in ("gabor", "lenia", "mlp", "gabor", "fourier"):
        ts.set_layout(layout(name))
    assert same(before, widths(ts))


def test_a_different_seed_gives_different_tiles():
    a = TournamentService(grid=4, layout=layout("gabor"), seed=0.5)
    a.init_population()
    b = TournamentService(grid=4, layout=layout("gabor"), seed=0.9)
    b.init_population()
    assert not same(widths(a), widths(b))


def test_two_services_at_one_seed_agree():
    a = TournamentService(grid=4, layout=layout("lenia"), seed=0.5)
    a.init_population()
    b = TournamentService(grid=4, layout=layout("lenia"), seed=0.5)
    b.init_population()
    assert same(widths(a), widths(b))


def test_changing_the_seed_reseeds_the_tiles_on_the_next_switch():
    """The seed is live: turning it changes what a switch lands on, which is
    what makes it a control rather than a constructor argument."""
    ts = TournamentService(grid=4, layout=layout("fourier"), seed=0.5)
    ts.init_population()
    before = widths(ts)
    ts.set_layout(layout("gabor"))
    ts.seed = 0.9
    ts.set_layout(layout("fourier"))
    assert not same(before, widths(ts))


def test_reset_still_offers_something_new():
    ts = TournamentService(grid=4, layout=layout("gabor"), seed=0.5)
    ts.init_population()
    first = widths(ts)
    ts.reset()
    assert not same(first, widths(ts))


def test_the_reset_sequence_replays_for_a_seed():
    def three(seed):
        ts = TournamentService(grid=4, layout=layout("gabor"), seed=seed)
        ts.init_population()
        out = [widths(ts)]
        for _ in range(2):
            ts.reset()
            out.append(widths(ts))
        return out

    for a, b in zip(three(0.5), three(0.5)):
        assert same(a, b)


@pytest.mark.parametrize("name", MODALITIES)
def test_the_tiles_are_drawn_from_the_same_prior_as_the_cohort_brains(name):
    """Same seed, same modality, same formula - the tournament's tiles and what
    "no rule loaded" puts on the canvas come from one generator."""
    lay = layout(name)
    ts = TournamentService(grid=2, layout=lay, seed=0.5)
    ts.init_population()
    want = generated_brains(lay, 0.5, ts.tiles)
    got = [np.asarray(g).reshape(-1) for g in ts.population]
    assert all(np.array_equal(x, y) for x, y in zip(got, want))
