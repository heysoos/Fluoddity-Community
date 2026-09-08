import pytest

from services.cohort_tiling import (
    MAX_COHORTS,
    cohort_of,
    cohorts_for,
    max_variants,
    tile_of,
)

ACTIVE = 100_000


@pytest.mark.parametrize("grid", [2, 3, 4, 5, 6, 7, 8])
def test_each_tile_contains_exactly_k_cohorts(grid):
    """Tile and cohort are both slices of the same particle numbering, so
    COHORTS must be a multiple of the tile count. If they were equal, every
    tile would hold one cohort - one mutation - and stay a monoculture."""
    tiles = grid * grid
    for k in range(1, max_variants(grid) + 1):
        cohorts = cohorts_for(grid, k)
        assert cohorts == k * tiles
        seen = {t: set() for t in range(tiles)}
        for i in range(0, ACTIVE, 7):
            seen[tile_of(i, ACTIVE, grid)].add(cohort_of(i, ACTIVE, cohorts))
        for t, s in seen.items():
            assert len(s) == k, f"grid={grid} k={k} tile={t} saw {len(s)} cohorts"


def test_the_monoculture_trap():
    """cohorts == tiles is the specific failure the feature must avoid."""
    grid, active = 4, ACTIVE
    seen = {t: set() for t in range(16)}
    for i in range(0, active, 7):
        seen[tile_of(i, active, grid)].add(cohort_of(i, active, 16))
    assert all(len(s) == 1 for s in seen.values())


@pytest.mark.parametrize(
    "grid,expected", [(2, 36), (3, 16), (4, 9), (5, 5), (6, 4), (7, 2), (8, 2)]
)
def test_max_variants_matches_the_144_cohort_cap(grid, expected):
    assert max_variants(grid) == expected
    assert cohorts_for(grid, expected) <= MAX_COHORTS


def test_tile_of_covers_every_tile_and_stays_in_range():
    for grid in (2, 4, 8):
        tiles = grid * grid
        got = {tile_of(i, ACTIVE, grid) for i in range(ACTIVE)}
        assert got == set(range(tiles))
    assert tile_of(ACTIVE - 1, ACTIVE, 4) == 15
    assert tile_of(ACTIVE, ACTIVE, 4) == 15, "must clamp, matching the shader"


# ---- boxing each cohort into its own cell --------------------------------

from services.cohort_tiling import box_grid  # noqa: E402


@pytest.mark.parametrize("cohorts", range(1, MAX_COHORTS + 1))
def test_every_cohort_gets_a_box(cohorts):
    gx, gy = box_grid(cohorts)
    assert gx * gy >= cohorts


@pytest.mark.parametrize("cohorts", range(1, MAX_COHORTS + 1))
def test_the_grid_is_as_square_as_it_can_be(cohorts):
    """A near-square layout, never a strip: 13 cohorts is 4x4 with three blank
    cells rather than 13x1."""
    gx, gy = box_grid(cohorts)
    assert 0 <= gx - gy <= 1


@pytest.mark.parametrize("cohorts,expected", [
    (1, (1, 1)), (2, (2, 1)), (4, (2, 2)), (6, (3, 2)), (12, (4, 3)),
    (13, (4, 4)), (64, (8, 8)), (144, (12, 12)),
])
def test_known_layouts(cohorts, expected):
    assert box_grid(cohorts) == expected


def test_no_box_is_wasted_beyond_one_row():
    """The blank cells never amount to a whole row - otherwise the grid could
    have been shorter."""
    for n in range(1, MAX_COHORTS + 1):
        gx, gy = box_grid(n)
        assert gx * (gy - 1) < n


def test_a_count_below_one_still_yields_a_box():
    assert box_grid(0) == (1, 1)
    assert box_grid(-5) == (1, 1)


@pytest.mark.parametrize("cohorts", [1, 2, 6, 12, 13, 64, 144])
def test_a_cohort_lands_in_exactly_one_box(cohorts):
    """Mirrors the shader: the box index IS the cohort index, so no cohort is
    split across two boxes and no box holds two cohorts."""
    gx, gy = box_grid(cohorts)
    boxes = {}
    for i in range(0, ACTIVE, 3):
        c = cohort_of(i, ACTIVE, cohorts)
        assert 0 <= c < gx * gy
        boxes.setdefault(c, set()).add(c)
    assert len(boxes) == cohorts
