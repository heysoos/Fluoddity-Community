"""Who owns the boxes, decided in one place.

The canvas is tiled by exactly one thing at a time: a tournament grid, or a box
per cohort, or nothing. Two owners would put the particles' seams and the
trails' seams in different places, which reads as a creature leaking into its
neighbour rather than as a bug.
"""
from sim import Sim
from state import SimState


class FakeSim(Sim):
    """Sim without a GL context; only the tiling bookkeeping is exercised."""

    def __init__(self):
        self._tournament_enabled = False
        self._tournament_grid = 4
        self._tournament_mutation = 0.0
        self._tournament_plain_colour = False
        self._cohort_boxes = False
        self._cohort_box_count = 64
        self._state = SimState()


def test_nothing_is_boxed_by_default():
    assert FakeSim().tile_state() == (0, (1, 1))


def test_a_tournament_boxes_by_particle_index():
    s = FakeSim()
    s.apply_tournament(True, grid=6, mutation=0.0)
    assert s.tile_state() == (1, (6, 6))


def test_cohort_boxes_get_a_grid_that_fits_the_count():
    s = FakeSim()
    s.apply_cohort_boxes(True, cohorts=12)
    assert s.tile_state() == (2, (4, 3))


def test_a_tournament_outranks_cohort_boxes():
    """Its tiles are already spoken for, and they carry the brains."""
    s = FakeSim()
    s.apply_cohort_boxes(True, cohorts=12)
    s.apply_tournament(True, grid=4, mutation=0.0)
    assert s.tile_state() == (1, (4, 4))


def test_turning_the_boxes_off_leaves_the_canvas_whole():
    s = FakeSim()
    s.apply_cohort_boxes(True, cohorts=9)
    s.apply_cohort_boxes(False, cohorts=9)
    assert s.tile_state() == (0, (1, 1))


def test_the_cohort_count_never_reaches_zero():
    """A zero count would ask for a grid with no cells; the shader divides by
    the box count."""
    s = FakeSim()
    s.apply_cohort_boxes(True, cohorts=0)
    mode, (gx, gy) = s.tile_state()
    assert mode == 2 and gx * gy >= 1
