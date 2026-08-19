"""Previews cost nothing when nobody is looking.

The row thumbnails only fill while the window is open, and inspect only
renders the one layer being looked at - a preview that ran unconditionally
would put the feature's cost back exactly where the design removed it.
"""
from __future__ import annotations

import pytest

moderngl = pytest.importorskip("moderngl")

from state.field_stack import FieldLayer, FieldStack  # noqa: E402
from services.field_sources import FrameContext  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 32
FRAME = FrameContext(time=0.0, frame_count=0, mouse=(0.5, 0.5),
                     prev_mouse=(0.5, 0.5), canvas_texture=None)


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


def a_layer():
    return FieldLayer(source="gradient", mapping="luminance",
                      destination="force", blend="replace",
                      params={"shape": 1, "falloff": 1.0, "centre": (0.5, 0.5)})


def rebuild(bus, layer):
    return bus.rebuild(FieldStack(layers=[layer]), RES, RES, 1.0, FRAME)


def test_no_thumbnail_when_previews_are_off(bus):
    layer = a_layer()
    bus.set_thumbnails_enabled(False)
    rebuild(bus, layer)
    assert bus.thumbnail_for(layer) is None


def test_a_thumbnail_appears_when_previews_are_on(bus):
    layer = a_layer()
    bus.set_thumbnails_enabled(True)
    rebuild(bus, layer)
    thumb = bus.thumbnail_for(layer)
    assert thumb is not None
    assert max(thumb.size) <= FieldBus.THUMB_MAX


def test_turning_previews_off_releases_the_thumbnails(bus):
    layer = a_layer()
    bus.set_thumbnails_enabled(True)
    rebuild(bus, layer)
    bus.set_thumbnails_enabled(False)
    assert bus.thumbnail_for(layer) is None


def test_inspect_source_returns_the_raw_source(bus):
    layer = a_layer()
    bus.set_thumbnails_enabled(True)
    rebuild(bus, layer)
    assert bus.inspect(layer, "source") is not None


def test_inspect_destination_shows_the_whole_field(bus):
    """A RENDERING of it, not the buffer: raw RG draws two opposite vectors
    as much the same colour, so the panel maps direction onto hue."""
    layer = a_layer()
    rebuild(bus, layer)
    shown = bus.inspect(layer, "destination")
    assert shown is not None
    assert shown is not bus.field_texture
    assert shown.size == bus.field_texture.size


def test_inspect_on_a_broken_layer_returns_none(bus):
    layer = FieldLayer(source="shader", params={"_file": "nope.frag"})
    rebuild(bus, layer)
    assert bus.inspect(layer, "source") is None


def test_a_dropped_layer_releases_its_thumbnail(bus):
    layer = a_layer()
    bus.set_thumbnails_enabled(True)
    rebuild(bus, layer)
    assert bus.thumbnail_for(layer) is not None
    bus.mark_dirty()
    bus.rebuild(FieldStack(layers=[]), RES, RES, 1.0, FRAME)
    assert bus.thumbnail_for(layer) is None
