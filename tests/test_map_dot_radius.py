"""The map's dots shrink only when they are actually crowding each other.

Zooming out puts the whole archive on one canvas, and at a few thousand entries
a fixed 3px dot turns the scatter into one solid mass - you can see there is a
lot of it and nothing else. The radius therefore follows how much ground the
dots cover, and the bar is the only tunable.
"""
import numpy as np
import pytest

from services import map_view

ORIGIN = (0.0, 0.0)
SIZE = (440.0, 320.0)          # the real map canvas at a typical panel width


def scatter(n, w=440.0, h=320.0, ox=0.0, oy=0.0, seed=0):
    """`n` points spread uniformly over a w x h patch at (ox, oy)."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(ox, ox + w, n).astype(np.float32),
            rng.uniform(oy, oy + h, n).astype(np.float32))


def radius(xs, ys):
    return map_view.dot_radius(xs, ys, ORIGIN, SIZE)


# -- the two ends -----------------------------------------------------------

@pytest.mark.parametrize("n", [1, 50, 200, 500])
def test_an_uncrowded_map_keeps_the_size_it_always_had(n):
    """The historical radius is the default and most maps never leave it."""
    assert radius(*scatter(n)) == map_view.DOT_RADIUS_MAX


def test_a_full_archive_shrinks(n=12000):
    assert radius(*scatter(n)) < map_view.DOT_RADIUS_MAX


def test_nothing_ever_goes_below_the_floor():
    """A dot under a pixel is not a smaller dot, it is noise."""
    huge = scatter(200_000, w=120.0, h=90.0, ox=160.0, oy=110.0)
    assert radius(*huge) == map_view.DOT_RADIUS_MIN


def test_an_empty_canvas_does_not_divide_by_zero():
    empty = (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
    assert radius(*empty) == map_view.DOT_RADIUS_MAX
    # Every point off the canvas is the same case, reached a different way.
    off = scatter(500, ox=10_000.0, oy=10_000.0)
    assert radius(*off) == map_view.DOT_RADIUS_MAX


# -- the property that makes it worth measuring ------------------------------

def test_a_dense_blob_shrinks_even_on_a_mostly_empty_canvas():
    """The load-bearing case, and the one a canvas-wide average would miss.

    Same point count; one spread over the whole canvas, one packed into a
    sixth of it. Coverage averaged over the whole canvas is identical for the
    two, so only a measure over the ground the points OCCUPY tells them apart.
    """
    spread_r = radius(*scatter(3000))
    blob_r = radius(*scatter(3000, w=180.0, h=130.0, ox=140.0, oy=100.0))
    assert blob_r < spread_r


def test_the_radius_only_falls_as_the_map_fills():
    counts = [200, 800, 2000, 5000, 12000, 20000]
    radii = [radius(*scatter(n)) for n in counts]
    assert radii == sorted(radii, reverse=True)
    assert radii[0] == map_view.DOT_RADIUS_MAX
    assert radii[-1] < radii[0]


def test_zooming_in_grows_the_dots_back():
    """Zoom culls, so the caller hands over fewer points spread over the same
    canvas - which is exactly the sparse case, and must return to full size."""
    crowded = radius(*scatter(9000))
    assert crowded < map_view.DOT_RADIUS_MAX
    # A 6x zoom leaves roughly 1/36 of them on screen.
    assert radius(*scatter(250)) == map_view.DOT_RADIUS_MAX


def test_the_bar_is_what_the_radius_is_solving_for():
    """Coverage of the occupied ground lands ON the bar once it is binding,
    which is what makes DOT_COVER_MAX the one number to tune."""
    xs, ys = scatter(9000)
    r = radius(xs, ys)
    assert map_view.DOT_RADIUS_MIN < r < map_view.DOT_RADIUS_MAX

    flat, _on, nx, ny, cell = map_view.bin_points(
        xs, ys, ORIGIN, SIZE, map_view.DOT_CELL_PX)
    ground = np.count_nonzero(np.bincount(flat, minlength=nx * ny)) * cell * cell
    cover = len(flat) * np.pi * r * r / ground
    assert cover == pytest.approx(map_view.DOT_COVER_MAX, rel=1e-6)


def test_a_lone_dot_per_cell_never_triggers_a_shrink():
    """DOT_CELL_PX is chosen against DOT_RADIUS_MAX so the sparse case cannot
    reach the bar. If someone widens the dot or narrows the cell, this is what
    says the two stopped agreeing."""
    one_dot = np.pi * map_view.DOT_RADIUS_MAX ** 2
    assert one_dot / (map_view.DOT_CELL_PX ** 2) < map_view.DOT_COVER_MAX
