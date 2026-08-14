"""The recording crop, on a real GPU.

canvas_view_rect returns a GL texture coordinate (v = 0 at the BOTTOM) while
tex_to_screen is top-down, and the two cancel exactly when the canvas is
centred - so a crop that is upside down looks perfect until someone pans. Every
rect here is deliberately off-centre.

Skipped when no GL context is available (CI, remote shells, no GPU).
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.record_view import RecordView  # noqa: E402

RES = 64
RED = (1.0, 0.0, 0.0, 1.0)      # bottom half, v < 0.5
GREEN = (0.0, 1.0, 0.0, 1.0)    # top half, v > 0.5


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context()
    except Exception as exc:                    # no GPU, no display, no driver
        pytest.skip(f"no GL context: {exc}")
    yield c
    c.release()


@pytest.fixture
def split_tex(ctx):
    """A texture whose bottom half is red and top half green, in GL order."""
    data = np.zeros((RES, RES, 4), dtype="f4")
    data[: RES // 2] = RED       # row 0 is v = 0, the bottom
    data[RES // 2:] = GREEN
    tex = ctx.texture((RES, RES), 4, data.tobytes(), dtype="f4")
    yield tex
    tex.release()


def _readback(tex):
    w, h = tex.size
    return np.frombuffer(tex.read(), dtype="f4").reshape(h, w, 4)


def test_cropping_the_top_half_returns_the_top_half(ctx, split_tex):
    """If v were flipped this would come back red, and a centred rect would
    never have told us."""
    view = RecordView(ctx, (RES, RES // 2))
    out = _readback(view.crop(split_tex, ((0.0, 0.5), (1.0, 1.0))))
    view.release()
    assert out[..., 1].min() > 0.9, "expected green - the crop is upside down"
    assert out[..., 0].max() < 0.1


def test_cropping_the_bottom_half_returns_the_bottom_half(ctx, split_tex):
    view = RecordView(ctx, (RES, RES // 2))
    out = _readback(view.crop(split_tex, ((0.0, 0.0), (1.0, 0.5))))
    view.release()
    assert out[..., 0].min() > 0.9, "expected red - the crop is upside down"
    assert out[..., 1].max() < 0.1


def test_an_identity_rect_reproduces_the_texture(ctx, split_tex):
    """The raw canvas views crop nothing, so this path must be lossless."""
    view = RecordView(ctx, (RES, RES))
    out = _readback(view.crop(split_tex, ((0.0, 0.0), (1.0, 1.0))))
    view.release()
    assert out[: RES // 2, :, 0].min() > 0.9
    assert out[RES // 2:, :, 1].min() > 0.9


def test_the_target_keeps_its_size_when_the_rect_moves(ctx, split_tex):
    """The rect is re-derived per frame but the encoder cannot take a size
    change, so a zoom must rescale into the frozen target."""
    view = RecordView(ctx, (RES, RES // 2))
    first = view.crop(split_tex, ((0.0, 0.5), (1.0, 1.0))).size
    second = view.crop(split_tex, ((0.25, 0.1), (0.75, 0.4))).size
    view.release()
    assert first == second == (RES, RES // 2)


def test_a_horizontal_crop_is_not_mirrored(ctx):
    """x needs no flip - window x and GL u both run left to right - so a bug
    here would be a silent left-right swap."""
    data = np.zeros((RES, RES, 4), dtype="f4")
    data[:, : RES // 2] = RED       # left half
    data[:, RES // 2:] = GREEN      # right half
    tex = ctx.texture((RES, RES), 4, data.tobytes(), dtype="f4")
    view = RecordView(ctx, (RES // 2, RES))
    out = _readback(view.crop(tex, ((0.5, 0.0), (1.0, 1.0))))
    view.release()
    tex.release()
    assert out[..., 1].min() > 0.9, "expected green - the crop is mirrored"
