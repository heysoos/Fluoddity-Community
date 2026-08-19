"""The trail destination has a target of its own.

Force and strafe are two channel pairs of ONE texture; the trail is not a
force at all - it is deposited into the sim's canvas, which the brains sense -
so it cannot share those channels and needs a second buffer. The bus rule
still holds over it: a destination nobody drives contributes exactly zero.

Whether the deposit reaches the canvas is not knowable here, because nothing
in this file runs sim.update. `python -m tools.drive_field_stack` does.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.field_sources import FrameContext  # noqa: E402
from state.field_stack import FieldLayer, FieldStack  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 32
FRAME = FrameContext(time=0.5, frame_count=3, mouse=(0, 0), prev_mouse=(0, 0),
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


def _run(bus, layers):
    bus.mark_dirty()
    bus.rebuild(FieldStack(layers=layers), RES, RES, 1.0, FRAME)


def _peak(tex, pair=0):
    if tex is None:
        return None
    data = np.frombuffer(tex.read(), dtype="f4").reshape(-1, 4)
    lo = 2 if pair else 0
    return float(np.abs(data[:, lo:lo + 2]).max())


def test_a_trail_layer_fills_the_trail_texture(bus):
    _run(bus, [FieldLayer(source="noise", mapping="luminance",
                          destination="trail")])
    assert bus.trail_texture is not None, "the trail destination had no target"
    assert _peak(bus.trail_texture) > 0.0, "the trail target came out empty"


def test_a_trail_layer_leaves_force_and_strafe_alone(bus):
    _run(bus, [FieldLayer(source="noise", mapping="luminance",
                          destination="trail")])
    assert _peak(bus.field_texture, 0) == 0.0, "it wrote the force field"
    assert _peak(bus.field_texture, 1) == 0.0, "it wrote the strafe field"


def test_a_force_layer_leaves_the_trail_unallocated(bus):
    """No layer drives it, so the sim must be handed None and skip the pass."""
    _run(bus, [FieldLayer(source="noise", destination="force")])
    assert bus.trail_texture is None


def test_disabling_the_only_trail_layer_releases_it(bus):
    layer = FieldLayer(source="noise", mapping="luminance", destination="trail")
    _run(bus, [layer])
    assert bus.trail_texture is not None
    layer.enabled = False
    _run(bus, [layer])
    assert bus.trail_texture is None, "a disabled layer left its field standing"


def test_the_global_switch_releases_the_trail_too(bus):
    stack = FieldStack(layers=[FieldLayer(source="noise", mapping="luminance",
                                          destination="trail")])
    bus.mark_dirty()
    bus.rebuild(stack, RES, RES, 1.0, FRAME)
    assert bus.trail_texture is not None
    stack.enabled = False
    bus.mark_dirty()
    bus.rebuild(stack, RES, RES, 1.0, FRAME)
    assert bus.trail_texture is None


def test_both_destinations_run_in_one_stack(bus):
    _run(bus, [FieldLayer(source="noise", destination="force"),
               FieldLayer(source="gradient", mapping="luminance",
                          destination="trail")])
    assert _peak(bus.field_texture, 0) > 0.0, "the force layer stopped working"
    assert _peak(bus.trail_texture) > 0.0, "the trail layer stopped working"


def test_the_trail_target_follows_the_bus_resolution(bus):
    layer = [FieldLayer(source="noise", mapping="luminance",
                        destination="trail")]
    _run(bus, layer)
    bus.mark_dirty()
    bus.rebuild(FieldStack(layers=layer), RES, RES, 0.5, FRAME)
    assert bus.trail_texture.size == bus.resolution
