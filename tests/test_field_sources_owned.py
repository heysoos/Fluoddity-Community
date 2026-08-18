"""Sources that own a texture return their own handle, costing no GPU pass.

The brush is the one source with memory, and that is the whole reason the bus
can be stateless: persistence belongs to a source, never to the bus.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from state.field_stack import FieldLayer  # noqa: E402
from services import field_sources  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 32


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
    b.ensure(RES, RES, 1.0)
    yield b
    b.cleanup()


def frame_with(canvas=None):
    return field_sources.FrameContext(
        time=0.0, frame_count=0, mouse=(0.5, 0.5),
        prev_mouse=(0.5, 0.5), canvas_texture=canvas)


def test_feedback_returns_the_canvas_texture_itself(ctx, bus):
    canvas = ctx.texture((8, 8), 2, dtype="f4")
    src = field_sources.make_source("feedback", ctx)
    got = src.evaluate(bus, FieldLayer(source="feedback"), frame_with(canvas))
    assert got is canvas, "feedback must borrow the canvas, not copy it"
    src.release()
    canvas.release()


def test_feedback_with_no_canvas_contributes_nothing(ctx, bus):
    src = field_sources.make_source("feedback", ctx)
    assert src.evaluate(bus, FieldLayer(source="feedback"), frame_with(None)) is None
    src.release()


def test_a_missing_image_file_sets_an_error_and_returns_none(ctx, bus):
    src = field_sources.make_source("image", ctx)
    layer = FieldLayer(source="image", params={"_file": "does_not_exist.png"})
    assert src.evaluate(bus, layer, frame_with()) is None
    assert src.error and "not found" in src.error.lower()
    src.release()


def test_a_shader_layer_with_no_file_sets_an_error(ctx, bus):
    src = field_sources.make_source("shader", ctx)
    assert src.evaluate(bus, FieldLayer(source="shader"), frame_with()) is None
    assert src.error
    src.release()


def test_a_shader_that_does_not_compile_reports_the_log(ctx, bus, tmp_path, monkeypatch):
    bad = tmp_path / "bad.frag"
    bad.write_text("#version 430\nout vec4 fragColor;\n"
                   "void main(){ fragColor = nope(1.0); }\n")
    monkeypatch.setattr(field_sources, "resolve_shader_path", lambda name: bad)
    src = field_sources.make_source("shader", ctx)
    layer = FieldLayer(source="shader", params={"_file": "bad.frag"})
    assert src.evaluate(bus, layer, frame_with()) is None
    assert src.error, "a compile failure must be reported, not swallowed"
    src.release()


def test_the_brush_owns_a_persistent_buffer(ctx, bus):
    src = field_sources.make_source("brush", ctx)
    data = np.full((RES, RES, 4), 0.25, dtype="f4")
    src.write(data)
    got = src.evaluate(bus, FieldLayer(source="brush"), frame_with())
    assert got is not None
    assert np.allclose(src.snapshot(), 0.25)
    src.release()


def test_the_brush_survives_being_evaluated_twice(ctx, bus):
    src = field_sources.make_source("brush", ctx)
    src.write(np.full((RES, RES, 4), 0.5, dtype="f4"))
    src.evaluate(bus, FieldLayer(source="brush"), frame_with())
    src.evaluate(bus, FieldLayer(source="brush"), frame_with())
    assert np.allclose(src.snapshot(), 0.5), "the bus must not clear a source's memory"
    src.release()
