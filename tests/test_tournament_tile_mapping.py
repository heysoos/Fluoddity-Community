"""Lock in the click->tile <-> shader tile-box orientation agreement.

The risky seam: command_handler maps a click (texture coords) to a tile index,
while entity_update.glsl maps a tile index to an entity-space box. If the two
disagree on the y direction, selecting a tile picks the vertically mirrored one.

Convention (verified against camera.screen_to_tex and services/entity_picker):
  tex_y == 0.0 is the BOTTOM of the canvas, and entity-space y == -half_extent
  is also the bottom. So tile 0 is bottom-left in both.
"""
import math

GRID = 4


def click_to_tile(tex_x, tex_y, grid=GRID):
    """Mirror of the mapping in command_handler._handle_mouse_clicks."""
    tx = min(grid - 1, max(0, int(tex_x * grid)))
    ty = min(grid - 1, max(0, int(tex_y * grid)))
    return ty * grid + tx


def tile_box(tile, canvas_w=1024, canvas_h=1024, grid=GRID):
    """Mirror of tournament_tile_box() in shaders/entity_update.glsl."""
    ca = canvas_w / canvas_h
    half = (math.sqrt(ca), 1.0 / math.sqrt(ca))
    tx = tile % grid
    ty = tile // grid
    cell = (2.0 * half[0] / grid, 2.0 * half[1] / grid)
    lo = (-half[0] + tx * cell[0], -half[1] + ty * cell[1])
    hi = (lo[0] + cell[0], lo[1] + cell[1])
    return lo, hi


def test_tile_zero_is_bottom_left_in_both_mappings():
    # A click near the bottom-left of the canvas selects tile 0...
    assert click_to_tile(0.05, 0.05) == 0
    # ...and tile 0's entity-space box is the bottom-left corner.
    lo, hi = tile_box(0)
    assert lo[0] < 0 and lo[1] < 0
    assert hi[0] < 0.01 and hi[1] < 0.01


def test_tile_15_is_top_right_in_both_mappings():
    assert click_to_tile(0.95, 0.95) == 15
    lo, hi = tile_box(15)
    assert lo[0] > -0.01 and lo[1] > -0.01
    assert hi[0] > 0 and hi[1] > 0


def test_row_major_ordering_matches():
    # Second column of the bottom row
    assert click_to_tile(0.30, 0.05) == 1
    # First column of the second row from the bottom
    assert click_to_tile(0.05, 0.30) == 4


def test_click_tile_and_box_agree_for_every_tile():
    """A click at the center of tile N's box must map back to tile N."""
    for tile in range(GRID * GRID):
        lo, hi = tile_box(tile)
        cx = (lo[0] + hi[0]) / 2
        cy = (lo[1] + hi[1]) / 2
        # entity space -> texture coords (inverse of get_can's uv mapping)
        half = (1.0, 1.0)  # square canvas => half_extent == (1,1)
        tex_x = cx / (2 * half[0]) + 0.5
        tex_y = cy / (2 * half[1]) + 0.5
        assert click_to_tile(tex_x, tex_y) == tile


def test_clicks_are_clamped_in_range():
    assert click_to_tile(-0.5, -0.5) == 0
    assert click_to_tile(1.5, 1.5) == 15
