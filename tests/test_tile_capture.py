"""Pins the readback contract: uint8 clamping, pack alignment, and the flip.
Run with: pytest -m gpu"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu

moderngl = pytest.importorskip("moderngl")

from services.tile_capture import TILE_PX, TileCapture  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")
    yield c
    c.release()


def _paint(ctx, grid, colors):
    """Fill each tile with its colour using scissored clears, with ty counted
    from the BOTTOM - i.e. in GL's own bottom-up viewport coordinates."""
    ctx.scissor = None
    for tile, col in enumerate(colors):
        tx, ty = tile % grid, tile // grid
        ctx.scissor = (tx * TILE_PX, ty * TILE_PX, TILE_PX, TILE_PX)
        ctx.clear(col[0] / 255, col[1] / 255, col[2] / 255, 1.0)
    ctx.scissor = None


@pytest.mark.parametrize("grid", [2, 4, 8])
def test_capture_returns_tiles_in_shader_order(ctx, grid):
    n = grid * grid
    colors = [((i * 7) % 200 + 20, (i * 13) % 200 + 20, (i * 29) % 200 + 20)
              for i in range(n)]
    cap = TileCapture(ctx, grid)
    tiles = cap.capture(lambda fbo: _paint(ctx, grid, colors))
    assert tiles.shape == (n, TILE_PX, TILE_PX, 3)
    for i, col in enumerate(colors):
        got = tuple(int(v) for v in tiles[i, TILE_PX // 2, TILE_PX // 2])
        assert got == col, f"tile {i}: got {got}, expected {col}"
    cap.release()


def test_capture_clamps_values_above_one(ctx):
    """An f4 attachment would store 4.0; f1 must clamp to 255."""
    cap = TileCapture(ctx, 2)
    tiles = cap.capture(lambda fbo: ctx.clear(4.0, 0.0, 0.0, 1.0))
    assert tiles[..., 0].max() == 255
    cap.release()


def test_resize_reallocates(ctx):
    cap = TileCapture(ctx, 2)
    assert cap.size == 2 * TILE_PX
    cap.resize(6)
    assert cap.size == 6 * TILE_PX
    assert cap.capture(None).shape[0] == 36
    cap.release()
