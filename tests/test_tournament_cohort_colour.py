"""Auto mode must not let cohort colouring decide the ranking.

Cohorts and tiles are both index-derived slices of the same particle numbering,
so with COHORTS a multiple of the tile count each tile owns a DISJOINT block of
cohort indices. color_by_cohort sets hue = hash(cohort), which makes each tile's
palette a fixed function of its slot and independent of its genome. CLIP weighs
colour heavily, so the slot decides a large share of the fitness.

Measured (16 tiles, one genome replicated into all of them, 6 genomes):
  colouring ON  -> tile identity explained 29.5% of spread, one tile won 5/6
  colouring OFF -> 9.9%, below the 16.7% expected from 6-sample noise alone

So Auto mode suppresses it, the same way it already owns MUTATION_SCALE.
Suppression happens at the uniform, never by writing to SimState, so the user's
saved appearance setting survives leaving Auto mode.
"""
import pytest

from sim import Sim
from state import SimState


class FakeSim(Sim):
    """Sim without a GL context; only the tournament bookkeeping is exercised."""

    def __init__(self):
        self._tournament_enabled = False
        self._tournament_grid = 4
        self._tournament_mutation = 0.0
        self._tournament_plain_colour = False
        self._state = SimState()


def test_apply_tournament_defaults_to_leaving_colour_alone():
    """Manual mode is a human looking at tiles; their palette choice stands."""
    s = FakeSim()
    s.apply_tournament(True, grid=4, mutation=0.0)
    assert s._tournament_plain_colour is False


def test_auto_mode_requests_cohort_colour_suppression():
    s = FakeSim()
    s.apply_tournament(True, grid=4, mutation=0.0, plain_colour=True)
    assert s._tournament_plain_colour is True


def test_suppression_never_writes_to_sim_state():
    """The user's saved appearance setting must survive Auto mode."""
    s = FakeSim()
    s._state.color_by_cohort = True
    s.apply_tournament(True, grid=4, mutation=0.0, plain_colour=True)
    assert s._state.color_by_cohort is True, "must not mutate the user's setting"


def test_disabling_the_tournament_clears_the_request():
    s = FakeSim()
    s.apply_tournament(True, grid=4, mutation=0.0, plain_colour=True)
    s.apply_tournament(False)
    assert s._tournament_plain_colour is False


@pytest.mark.gpu
def test_uniform_is_actually_forced_off_on_the_gpu():
    """The bookkeeping above is worthless if the uniform still says True."""
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")

    sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
    st = SimState()
    st.color_by_cohort = True
    sim.apply_state(st)

    # moderngl reports a bool uniform as int 1/0.
    sim.apply_tournament(True, grid=4, mutation=0.0, plain_colour=False)
    sim.entity_update(ctx)
    assert bool(sim.entity_update_program['COLOR_BY_COHORT'].value) is True

    sim.apply_tournament(True, grid=4, mutation=0.0, plain_colour=True)
    sim.entity_update(ctx)
    assert bool(sim.entity_update_program['COLOR_BY_COHORT'].value) is False
    assert st.color_by_cohort is True, "user's saved setting must be untouched"

    ctx.release()
