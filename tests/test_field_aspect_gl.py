"""A source shaped differently from the bus is cropped, never squeezed.

A 4:3 camera stretched into a square bus is the visible symptom. The test uses
a source with a CIRCLE in it, because a stretched circle and a cropped circle
differ in shape while a stretched gradient and a cropped gradient can both look
plausible.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.field_sources import FrameContext  # noqa: E402
from state.field_stack import FieldLayer, FieldStack  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 64
FRAME = FrameContext(time=0.0, frame_count=0, mouse=(0, 0), prev_mouse=(0, 0),
                     canvas_texture=None)


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


@pytest.fixture
def bus(ctx):
    b = FieldBus(ctx)
    yield b
    b.cleanup()


def _circle_texture(ctx, width, height):
    """A white disc on black, as wide as it is tall in PIXELS."""
    ys, xs = np.mgrid[0:height, 0:width]
    radius = min(width, height) * 0.3
    inside = ((xs - width / 2) ** 2 + (ys - height / 2) ** 2) < radius ** 2
    data = np.where(inside, 255, 0).astype(np.uint8)
    return ctx.texture((width, height), 3,
                       np.repeat(data[:, :, None], 3, axis=2).tobytes())


class _FixedSource:
    error = None
    animated = False

    def __init__(self, tex):
        self._tex = tex

    def evaluate(self, bus, layer, frame):
        return self._tex

    def release(self):
        pass


def _render(bus, tex):
    layer = FieldLayer(source="image", mapping="luminance", blend="replace")
    bus.ensure(RES, RES, 1.0)
    bus._sources[layer.uid] = ("image", _FixedSource(tex))
    bus.mark_dirty()
    bus.rebuild(FieldStack(layers=[layer]), RES, RES, 1.0, FRAME)
    return np.frombuffer(bus.field_texture.read(),
                         dtype="f4").reshape(RES, RES, 4)[..., 0]


def _extent(mask, axis):
    hit = np.where(mask.any(axis=axis))[0]
    return 0 if len(hit) == 0 else hit[-1] - hit[0] + 1


def test_a_wide_source_keeps_its_shape(ctx, bus):
    """4:3 into a square bus: the disc must stay round, not become an oval."""
    field = _render(bus, _circle_texture(ctx, 640, 480))
    mask = field > 0.5 * field.max()
    across = _extent(mask, axis=0)
    down = _extent(mask, axis=1)
    assert across > 0 and down > 0, "nothing was drawn"
    assert abs(across - down) <= 2, (
        f"the disc came out {across} wide and {down} tall - it was stretched")


def test_a_tall_source_keeps_its_shape(ctx, bus):
    field = _render(bus, _circle_texture(ctx, 480, 640))
    mask = field > 0.5 * field.max()
    assert abs(_extent(mask, 0) - _extent(mask, 1)) <= 2


def test_a_square_source_is_untouched(ctx, bus):
    """The procedural sources render at the bus size and must not be resampled."""
    field = _render(bus, _circle_texture(ctx, RES, RES))
    mask = field > 0.5 * field.max()
    assert abs(_extent(mask, 0) - _extent(mask, 1)) <= 1


def test_the_crop_fills_the_frame(ctx, bus):
    """Cover, not contain: no blank margin where the source ran out."""
    ones = np.full((480, 640, 3), 255, dtype=np.uint8)
    field = _render(bus, ctx.texture((640, 480), 3, ones.tobytes()))
    assert field.min() > 0.0, "the source did not cover the whole field"
