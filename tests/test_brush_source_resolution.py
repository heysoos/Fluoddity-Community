"""A brush buffer is never asked for a zero-sized texture.

`_brush_target` hands `field_bus.resolution` straight to `_BrushSource.ensure`,
and a bus that has not sized itself reports (0, 0) - which is its documented
resting state, not a corrupt one: `__init__` sets it and `_release_target` puts
it back. The bus's own `ensure` guards with max(1, ...); the brush's did not, so
a stroke aimed at an empty stack built a 0x0 texture and moderngl refused the
framebuffer with "invalid color attachment", taking the app down on the first
click after Advanced Drawing was switched on.
"""
from __future__ import annotations

import pytest

moderngl = pytest.importorskip("moderngl")

from services import field_sources  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


def test_a_fresh_bus_rests_at_zero_resolution(ctx):
    """The state that reaches the brush. If this changes, the guard below
    is guarding nothing and the real fix moved elsewhere."""
    bus = FieldBus(ctx)
    try:
        assert bus.resolution == (0, 0)
    finally:
        bus.cleanup()


def test_the_brush_survives_a_bus_that_has_not_sized_itself(ctx):
    """The crash, exactly as _brush_target reaches it."""
    bus = FieldBus(ctx)
    src = field_sources.make_source("brush", ctx)
    try:
        tex = src.ensure(*bus.resolution)
        assert tex is not None
        assert tex.width >= 1 and tex.height >= 1
        assert src.framebuffer is not None
    finally:
        src.release()
        bus.cleanup()


def test_a_released_bus_rests_at_zero_too(ctx):
    """_release_target puts _res back to (0, 0), so an emptied stack reaches
    the brush the same way a fresh one does."""
    bus = FieldBus(ctx)
    src = field_sources.make_source("brush", ctx)
    try:
        bus.ensure(64, 64, 0.5)
        assert bus.resolution == (32, 32)
        bus.rebuild(None, 64, 64, 0.5, None)
        tex = src.ensure(*bus.resolution)
        assert tex.width >= 1 and tex.height >= 1
    finally:
        src.release()
        bus.cleanup()
