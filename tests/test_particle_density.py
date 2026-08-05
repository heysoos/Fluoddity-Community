"""Particle density: how crowded the world is, independent of how big it is.

world_size scales entity count and canvas area by the same factor, so density
is invariant under it - 600000*ws particles over (1024*sqrt(ws))^2 texels is
always 0.572 particles/texel. This multiplier is the knob that actually moves it.
"""
import math

import pytest

from services.cohort_tiling import cohort_of, tile_of
from state.preferences_state import PreferencesState


BASE = 600000
CANVAS = 1024


# -- the invariance this feature exists to break ----------------------------


@pytest.mark.parametrize("ws", [0.02, 0.4, 1.0, 4.0])
def test_world_size_alone_cannot_change_density(ws):
    """The premise: ws cancels out, so the World Size slider is not a density
    control no matter where it is put."""
    n = int(BASE * ws)
    side = int(CANVAS * math.sqrt(ws))
    # Exact but for the integer truncation of the canvas side, which bites
    # hardest on the smallest worlds.
    assert n / (side * side) == pytest.approx(0.572, rel=0.02)


def test_density_multiplier_scales_particles_but_not_the_canvas():
    from sim import Sim

    sparse = Sim.__new__(Sim)
    sparse.world_size, sparse.particle_density = 4.0, 0.1
    sparse.canvas_aspect_ratio = "1:1"

    dense = Sim.__new__(Sim)
    dense.world_size, dense.particle_density = 4.0, 1.0
    dense.canvas_aspect_ratio = "1:1"

    assert sparse.get_canvas_dimensions() == dense.get_canvas_dimensions()
    assert sparse.get_entity_count() * 10 == dense.get_entity_count()


def test_density_one_reproduces_the_old_entity_count():
    """Existing saved preferences must not change what they render."""
    from sim import Sim

    for ws in (0.02, 0.4, 1.0, 4.0):
        s = Sim.__new__(Sim)
        s.world_size, s.particle_density = ws, 1.0
        s.canvas_aspect_ratio = "1:1"
        assert s.get_entity_count() == int(BASE * ws)


def test_a_tiny_density_still_leaves_particles_to_dispatch():
    """A zero-length buffer is a GL error, and one workgroup is the floor."""
    from sim import Sim

    s = Sim.__new__(Sim)
    s.world_size, s.particle_density = 0.02, 0.0
    s.canvas_aspect_ratio = "1:1"
    assert s.get_entity_count() >= 1


def test_default_preference_is_the_current_behaviour():
    assert PreferencesState().particle_density == 1.0


# -- the trap: index-derived slices must use the REAL count ------------------


@pytest.mark.parametrize("density", [1.0, 0.5, 0.1])
def test_tiles_still_span_the_whole_grid_at_any_density(density):
    """tile_of divides index by the active count. If that denominator kept
    using 600000*world_size while the buffer shrank, indices would only reach
    `density` of the way up and the top tiles would render empty."""
    active = int(BASE * 0.4 * density)
    grid = 4
    seen = {tile_of(i, active, grid) for i in range(0, active, max(1, active // 5000))}
    assert seen == set(range(grid * grid))


@pytest.mark.parametrize("density", [1.0, 0.5, 0.1])
def test_cohorts_still_span_their_full_range_at_any_density(density):
    active = int(BASE * 0.4 * density)
    cohorts = 32
    seen = {cohort_of(i, active, cohorts)
            for i in range(0, active, max(1, active // 5000))}
    assert seen == set(range(cohorts))


def test_a_stale_denominator_would_compress_tiles_into_the_bottom_half():
    """Documents the failure mode above, so the guard tests cannot be
    'fixed' by making them vacuous."""
    real_active = int(BASE * 0.4 * 0.5)      # buffer actually halved
    stale = int(BASE * 0.4)                  # denominator not updated
    seen = {tile_of(i, stale, 4) for i in range(real_active)}
    assert seen == set(range(8)), "only the bottom 8 of 16 tiles get particles"
