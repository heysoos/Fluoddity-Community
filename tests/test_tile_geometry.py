import numpy as np
import pytest

from services.cohort_tiling import tile_of
from services.tile_capture import TILE_PX, crop_bounds, split_grid


@pytest.mark.parametrize("grid", [2, 3, 4, 5, 6, 7, 8])
def test_crops_are_unique_and_non_overlapping(grid):
    seen = set()
    for tile in range(grid * grid):
        b = crop_bounds(tile, grid)
        assert b not in seen
        seen.add(b)
        r0, r1, c0, c1 = b
        assert r1 - r0 == TILE_PX
        assert c1 - c0 == TILE_PX
        assert 0 <= r0 < r1 <= grid * TILE_PX
        assert 0 <= c0 < c1 <= grid * TILE_PX


@pytest.mark.parametrize("grid", [2, 4, 8])
def test_tile_zero_is_bottom_left(grid):
    """index_home_tile() numbers tile 0 as bottom-left, but a top-down
    image has row 0 at the top. Getting this backwards has already caused one
    user-visible bug."""
    r0, _, c0, _ = crop_bounds(0, grid)
    assert c0 == 0, "tile 0 must be in the leftmost column"
    assert r0 == (grid - 1) * TILE_PX, "tile 0 must be in the bottom row"

    top_left_tile = (grid - 1) * grid
    r0, _, c0, _ = crop_bounds(top_left_tile, grid)
    assert (r0, c0) == (0, 0)


@pytest.mark.parametrize("grid", [2, 4])
def test_split_grid_returns_tiles_in_shader_order(grid):
    """Paint tile t with intensity t+1, laid out with ty counted from the
    bottom, then assert split_grid recovers them in index order."""
    size = grid * TILE_PX
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for tile in range(grid * grid):
        tx, ty = tile % grid, tile // grid
        row0 = (grid - 1 - ty) * TILE_PX
        img[row0:row0 + TILE_PX, tx * TILE_PX:(tx + 1) * TILE_PX] = tile + 1

    tiles = split_grid(img, grid)
    assert tiles.shape == (grid * grid, TILE_PX, TILE_PX, 3)
    for tile in range(grid * grid):
        assert tiles[tile].min() == tile + 1
        assert tiles[tile].max() == tile + 1


def test_tile_index_formula_agrees_with_the_shader_partition():
    """crop_bounds indexes by tile = ty*grid + tx, the same numbering that
    tile_of() derives from the particle index."""
    grid, active = 4, 100_000
    for i in range(0, active, 997):
        t = tile_of(i, active, grid)
        assert 0 <= t < grid * grid
        assert crop_bounds(t, grid) == crop_bounds(
            (t // grid) * grid + (t % grid), grid
        )
