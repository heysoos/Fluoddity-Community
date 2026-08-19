"""The rebuild loop: order, the disabled-layer guarantee, and the dirty flag.

The disabled-layer test is the regression test for the reported defect - a
shader-driven field that could not be turned off, because nothing cleared the
texture it had painted.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from state.field_stack import FieldLayer, FieldStack  # noqa: E402
from services.field_sources import FrameContext  # noqa: E402
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
    yield b
    b.cleanup()


FRAME = FrameContext(time=0.0, frame_count=0, mouse=(0.5, 0.5),
                     prev_mouse=(0.5, 0.5), canvas_texture=None)


def rebuild(bus, stack):
    return bus.rebuild(stack, RES, RES, 1.0, FRAME)


def read(bus):
    tex = bus.field_texture
    return np.frombuffer(tex.read(), dtype="f4").reshape(RES, RES, 4)


def gradient_layer(**kw):
    kw.setdefault("source", "gradient")
    kw.setdefault("mapping", "luminance")
    kw.setdefault("destination", "force")
    kw.setdefault("blend", "replace")
    kw.setdefault("params", {"shape": 1, "falloff": 1.0, "centre": (0.5, 0.5)})
    return FieldLayer(**kw)


def test_an_empty_stack_allocates_nothing(bus):
    assert rebuild(bus, FieldStack()) is False
    assert bus.field_texture is None
    assert bus.pass_count == 0


def test_a_disabled_layer_contributes_exactly_zero(bus):
    rebuild(bus, FieldStack(layers=[gradient_layer()]))
    assert read(bus)[..., 0].max() > 0.0

    layer = gradient_layer(enabled=False)
    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[layer]))
    assert np.allclose(read(bus), 0.0), "a disabled layer left its contribution behind"


def test_removing_the_last_layer_clears_the_field(bus):
    rebuild(bus, FieldStack(layers=[gradient_layer()]))
    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[]))
    assert bus.field_texture is None or np.allclose(read(bus), 0.0)


def test_stack_order_decides_the_winner(bus):
    first = gradient_layer(strength=1.0)
    second = gradient_layer(strength=0.25)
    rebuild(bus, FieldStack(layers=[first, second]))
    late = read(bus)[..., 0].max()

    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[second, first]))
    early = read(bus)[..., 0].max()

    assert late < early, "the later replace layer must win"


def test_a_clean_bus_does_no_work(bus):
    stack = FieldStack(layers=[gradient_layer()])
    assert rebuild(bus, stack) is True
    assert bus.dirty is False
    assert rebuild(bus, stack) is False


def test_the_pass_count_survives_a_clean_frame(bus):
    """It counts the last REBUILD, and the readout is on screen every frame.

    Zeroing it made the window say "0 passes" for every frame a layer was
    quietly forcing, which reads as a stack that is doing nothing.
    """
    stack = FieldStack(layers=[gradient_layer()])
    rebuild(bus, stack)
    ran = bus.pass_count
    assert ran > 0
    rebuild(bus, stack)
    assert bus.pass_count == ran


def test_marking_dirty_makes_it_rebuild(bus):
    stack = FieldStack(layers=[gradient_layer()])
    rebuild(bus, stack)
    bus.mark_dirty()
    assert rebuild(bus, stack) is True


def test_a_resolution_change_forces_a_rebuild(bus):
    stack = FieldStack(layers=[gradient_layer()])
    rebuild(bus, stack)
    assert bus.rebuild(stack, RES, RES, 0.5, FRAME) is True
    assert bus.resolution == (RES // 2, RES // 2)


def test_a_broken_layer_does_not_stop_the_others(bus):
    broken = FieldLayer(source="shader", params={"_file": "nope.frag"},
                        destination="force")
    good = gradient_layer()
    rebuild(bus, FieldStack(layers=[broken, good]))
    assert broken.error, "the broken layer must carry its own error"
    assert good.error is None
    assert read(bus)[..., 0].max() > 0.0, "the healthy layer stopped rendering"


def test_an_error_clears_when_the_layer_is_fixed(bus):
    layer = FieldLayer(source="shader", params={"_file": "nope.frag"})
    rebuild(bus, FieldStack(layers=[layer]))
    assert layer.error

    fixed = gradient_layer(uid=layer.uid)
    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[fixed]))
    assert fixed.error is None


def test_gpu_state_is_keyed_by_uid_not_by_position(bus):
    a = gradient_layer()
    b = gradient_layer()
    rebuild(bus, FieldStack(layers=[a, b]))
    source_a = bus.source_for(a)

    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[b, a]))
    assert bus.source_for(a) is source_a, "reordering rebuilt a source from scratch"


def test_a_source_dropped_from_the_stack_is_released(bus):
    layer = gradient_layer()
    rebuild(bus, FieldStack(layers=[layer]))
    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[]))
    assert bus.source_for(layer) is None


def test_pass_count_reports_what_ran(bus):
    rebuild(bus, FieldStack(layers=[gradient_layer(), gradient_layer()]))
    # Two procedural layers: a source pass and a composite pass each.
    assert bus.pass_count == 4


def test_an_animated_source_rebuilds_without_being_marked_dirty(bus):
    """Noise reads the clock, so it must advance on its own.

    Left to the dirty flag it froze on whichever frame a slider was last
    touched - a still image in the thumbnail, the inspector and the sim.
    """
    stack = FieldStack(layers=[FieldLayer(source="noise")])
    assert rebuild(bus, stack) is True
    assert bus.dirty is False
    assert rebuild(bus, stack) is True, "an animated source stopped redrawing"


def test_a_static_source_still_goes_quiet(bus):
    """The fast path has to survive: a stack that cannot change is free."""
    layer = FieldLayer(source="image")
    layer.params["_file"] = ""
    stack = FieldStack(layers=[layer])
    rebuild(bus, stack)
    assert rebuild(bus, stack) is False


def test_a_disabled_animated_layer_does_not_keep_the_bus_awake(bus):
    layer = FieldLayer(source="noise", enabled=False)
    stack = FieldStack(layers=[layer])
    rebuild(bus, stack)
    assert rebuild(bus, stack) is False


def test_the_scratch_is_not_left_on_a_mipmap_filter(bus):
    """A mipmap min-filter with no chain samples as BLACK, which is what the
    thumbnail and the inspector draw unless Blur happens to build one."""
    import moderngl
    rebuild(bus, FieldStack(layers=[FieldLayer(source="noise", blur=0.0)]))
    assert bus.scratch_texture.filter == (moderngl.LINEAR, moderngl.LINEAR)
    rebuild(bus, FieldStack(layers=[FieldLayer(source="noise", blur=3.0)]))
    assert bus.scratch_texture.filter == (moderngl.LINEAR, moderngl.LINEAR)
