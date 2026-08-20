"""Noise Speed must move the field through TIME, not across the canvas.

The first version added `time * speed` to the y coordinate, so the pattern
slid upward: every feature kept its shape and simply translated. A source-level
reading cannot tell that from evolving in place, so this renders two frames and
asks whether the later one is a shifted copy of the earlier.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.field_sources import FrameContext  # noqa: E402
from state.field_stack import FieldLayer, FieldStack  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 64


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


def frame_at(t):
    return FrameContext(time=t, frame_count=int(t * 60), mouse=(0.0, 0.0),
                        prev_mouse=(0.0, 0.0), canvas_texture=None)


def render(bus, t, **params):
    layer = FieldLayer(source="noise", mapping="luminance", blend="replace",
                       params={"scale": 4.0, "octaves": 1, "warp": 0.0,
                               **params})
    bus.mark_dirty()
    bus.rebuild(FieldStack(layers=[layer]), RES, RES, 1.0, frame_at(t))
    return np.frombuffer(bus.field_texture.read(),
                         dtype="f4").reshape(RES, RES, 4)[..., 0].copy()


def test_speed_changes_the_field_over_time(bus):
    a = render(bus, 0.0, speed=1.0)
    b = render(bus, 1.0, speed=1.0)
    assert not np.allclose(a, b), "Speed did nothing"


def test_speed_zero_holds_the_field_still(bus):
    a = render(bus, 0.0, speed=0.0)
    b = render(bus, 5.0, speed=0.0)
    assert np.allclose(a, b), "the field moved with Speed at zero"


def test_time_does_not_merely_scroll_the_pattern(bus):
    """The defect: `p.y += time*speed` translates rather than evolves.

    A translated copy matches itself at some row offset far better than an
    evolved one does, so the best match over all vertical shifts is what
    separates the two.
    """
    a = render(bus, 0.0, speed=1.0)
    b = render(bus, 0.35, speed=1.0)

    def difference(shift):
        return float(np.abs(np.roll(a, shift, axis=0) - b).mean())

    unshifted = difference(0)
    best_shift = min((difference(s), s) for s in range(1, RES))
    assert best_shift[0] > 0.35 * unshifted, (
        f"a shift of {best_shift[1]} rows recovered the earlier frame "
        f"({best_shift[0]:.4f} against {unshifted:.4f} unshifted) - the "
        "pattern is sliding, not evolving")
