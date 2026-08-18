"""The brush's buffer is the field handler's target, and legacy PNGs still load.

Every preset on disk predates the stack. One with a companion _fields.png must
come back as two brush layers over that texture, or its field silently
vanishes.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from services.config_saver import legacy_brush_stack
from services.field_handler import FieldHandler, ensure_brush_layer
from state.field_stack import FieldStack, stack_from_dict


class FakeBrush:
    def __init__(self):
        self.data = None

    def snapshot(self):
        return self.data

    def write(self, data):
        self.data = np.array(data, dtype="f4")

    def clear(self):
        self.data = None


class FakeBus:
    def __init__(self, brush):
        self._brush = brush
        self.dirtied = False

    def brush_source(self):
        return self._brush

    def ensure_brush_source(self, stack):
        return self._brush

    def mark_dirty(self):
        self.dirtied = True


def make_ui(stack=None):
    return SimpleNamespace(
        preferences=SimpleNamespace(force_field_strength=1.0,
                                    strafe_field_strength=1.0),
        field_stack=stack if stack is not None else FieldStack())


@pytest.fixture
def handler():
    brush = FakeBrush()
    bus = FakeBus(brush)
    sim = SimpleNamespace(get_canvas_dimensions=lambda: (8, 8))
    return FieldHandler(bus, sim), brush, bus


def test_a_snapshot_reads_the_brush_buffer(handler):
    h, brush, _ = handler
    brush.write(np.full((8, 8, 4), 0.3, dtype="f4"))
    data, strengths = h.snapshot_with_strengths(make_ui())
    assert data is not None
    assert np.allclose(data, 0.3)
    assert strengths == (1.0, 1.0)


def test_a_zero_buffer_snapshots_as_none(handler):
    h, brush, _ = handler
    brush.write(np.zeros((8, 8, 4), dtype="f4"))
    data, _ = h.snapshot_with_strengths(make_ui())
    assert data is None


def test_a_legacy_config_installs_two_brush_layers(handler):
    h, brush, bus = handler
    ui = make_ui()
    config = SimpleNamespace(field_stack={}, force_field_strength=1.0,
                             strafe_field_strength=1.0)
    h.cache.get = lambda *a, **k: np.full((8, 8, 4), 0.4, dtype="f4")

    h.apply_for_config(config, "whatever.json", ui)

    assert [l.source for l in ui.field_stack.layers] == ["brush", "brush"]
    assert np.allclose(brush.data, 0.4)
    assert bus.dirtied


def test_a_config_with_its_own_stack_is_used_verbatim(handler):
    h, _, _ = handler
    ui = make_ui()
    own = {"layers": [{"uid": "x", "source": "noise", "destination": "force"}]}
    config = SimpleNamespace(field_stack=own, force_field_strength=1.0,
                             strafe_field_strength=1.0)
    h.cache.get = lambda *a, **k: None

    h.apply_for_config(config, "whatever.json", ui)

    assert [l.source for l in ui.field_stack.layers] == ["noise"]


def test_a_config_with_neither_leaves_an_empty_stack(handler):
    h, _, _ = handler
    ui = make_ui(stack_from_dict(legacy_brush_stack()))
    config = SimpleNamespace(field_stack={}, force_field_strength=1.0,
                             strafe_field_strength=1.0)
    h.cache.get = lambda *a, **k: None

    h.apply_for_config(config, "whatever.json", ui)

    assert ui.field_stack.layers == []


def test_drawing_at_a_destination_with_no_brush_layer_adds_one():
    """Selecting Force Field and painting must still put paint somewhere."""
    stack = FieldStack()
    ensure_brush_layer(stack, "force")
    assert [(l.source, l.destination) for l in stack.layers] == [("brush", "force")]


def test_ensuring_a_brush_layer_twice_adds_only_one():
    stack = FieldStack()
    ensure_brush_layer(stack, "force")
    ensure_brush_layer(stack, "force")
    assert len(stack.layers) == 1


def test_force_and_strafe_get_their_own_brush_layers():
    stack = FieldStack()
    ensure_brush_layer(stack, "force")
    ensure_brush_layer(stack, "strafe")
    assert {l.destination for l in stack.layers} == {"force", "strafe"}
    channels = {l.destination: l.params.get("_channels") for l in stack.layers}
    assert channels == {"force": "xy", "strafe": "zw"}
