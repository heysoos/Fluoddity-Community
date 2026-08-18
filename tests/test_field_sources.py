"""The source registry: descriptors are data, evaluation returns a texture.

Two properties matter beyond "it renders". A source returns a HANDLE rather
than filling a caller's buffer, so a texture-owning source costs no GPU pass.
And params_for() merges a shader's discovered uniforms with the built-ins, so
a .frag layer's UI comes from the file rather than from a hardcoded list.
"""
from __future__ import annotations

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


@pytest.fixture
def frame():
    return field_sources.FrameContext(
        time=1.0, frame_count=60, mouse=(0.5, 0.5),
        prev_mouse=(0.5, 0.5), canvas_texture=None)


def test_every_descriptor_has_a_unique_key():
    keys = [d.key for d in field_sources.descriptors()]
    assert len(keys) == len(set(keys))


def test_the_stage_one_sources_are_registered():
    keys = {d.key for d in field_sources.descriptors()}
    assert {"noise", "image", "gradient", "shader", "brush", "feedback"} <= keys


def test_only_some_sources_keep_history():
    assert field_sources.get("shader").keeps_history
    assert field_sources.get("feedback").keeps_history
    assert not field_sources.get("image").keeps_history
    assert not field_sources.get("gradient").keeps_history


@pytest.mark.parametrize("key", ["noise", "gradient"])
def test_a_procedural_source_returns_a_texture_at_bus_resolution(ctx, bus, frame, key):
    src = field_sources.make_source(key, ctx)
    tex = src.evaluate(bus, FieldLayer(source=key), frame)
    assert tex is not None
    assert tex.size == (RES, RES)
    src.release()


def test_a_procedural_source_reports_no_error(ctx, bus, frame):
    src = field_sources.make_source("noise", ctx)
    src.evaluate(bus, FieldLayer(source="noise"), frame)
    assert src.error is None
    src.release()


def test_builtin_params_are_exposed_for_the_ui(ctx):
    names = [p.name for p in field_sources.params_for(FieldLayer(source="noise"))]
    assert "scale" in names
    assert "speed" in names


def test_a_shader_layer_with_no_file_has_no_params():
    layer = FieldLayer(source="shader", params={})
    assert field_sources.params_for(layer) == []
