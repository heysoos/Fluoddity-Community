"""The composite pass, on a real GPU.

Blending is native GL state rather than a shader branch, and a destination is
restricted to its own channel pair by a colour mask. Both are easy to get
subtly wrong and invisible on screen: a force layer that also writes strafe
looks like a preset that strafes more than it used to.

Skipped when no GL 4.3 context is available.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from state.field_stack import FieldLayer  # noqa: E402
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


def flat(ctx, value):
    """A RES x RES RGBA32F texture with every texel set to `value`."""
    data = np.tile(np.array(value, dtype="f4"), (RES, RES, 1))
    return ctx.texture((RES, RES), 4, data.tobytes(), dtype="f4")


def read(bus):
    tex = bus.field_texture
    return np.frombuffer(tex.read(), dtype="f4").reshape(RES, RES, 4)


def layer(**kw):
    kw.setdefault("mapping", "rg_direct")
    kw.setdefault("destination", "force")
    kw.setdefault("blend", "replace")
    return FieldLayer(**kw)


def test_replace_writes_the_value(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.25, 0.5, 0.0, 0.0)), layer(strength=1.0))
    out = read(bus)
    assert np.allclose(out[..., 0], 0.25)
    assert np.allclose(out[..., 1], 0.5)


def test_strength_scales_the_value(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.4, 0.0, 0.0, 0.0)), layer(strength=0.5))
    assert np.allclose(read(bus)[..., 0], 0.2)


def test_add_accumulates(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.3, 0.0, 0.0, 0.0)), layer(blend="add"))
    bus.composite_one(flat(ctx, (0.2, 0.0, 0.0, 0.0)), layer(blend="add"))
    assert np.allclose(read(bus)[..., 0], 0.5)


def test_multiply_multiplies(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.5, 0.0, 0.0, 0.0)), layer(blend="replace"))
    bus.composite_one(flat(ctx, (0.5, 0.0, 0.0, 0.0)), layer(blend="multiply"))
    assert np.allclose(read(bus)[..., 0], 0.25)


def test_max_takes_the_larger(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.7, 0.0, 0.0, 0.0)), layer(blend="replace"))
    bus.composite_one(flat(ctx, (0.2, 0.0, 0.0, 0.0)), layer(blend="max"))
    assert np.allclose(read(bus)[..., 0], 0.7)


def test_a_force_layer_leaves_strafe_untouched(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.9, 0.9, 0.0, 0.0)),
                      layer(destination="strafe"))
    bus.composite_one(flat(ctx, (0.1, 0.1, 0.0, 0.0)),
                      layer(destination="force"))
    out = read(bus)
    assert np.allclose(out[..., 2], 0.9), "strafe was clobbered by a force layer"
    assert np.allclose(out[..., 3], 0.9)
    assert np.allclose(out[..., 0], 0.1)


def test_a_strafe_layer_leaves_force_untouched(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.9, 0.9, 0.0, 0.0)),
                      layer(destination="force"))
    bus.composite_one(flat(ctx, (0.1, 0.1, 0.0, 0.0)),
                      layer(destination="strafe"))
    out = read(bus)
    assert np.allclose(out[..., 0], 0.9), "force was clobbered by a strafe layer"
    assert np.allclose(out[..., 2], 0.1)


def test_a_nan_source_reaches_the_destination_as_zero(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (np.nan, np.inf, 0.0, 0.0)), layer())
    out = read(bus)
    assert np.all(np.isfinite(out)), "a non-finite value escaped the composite"
    assert np.allclose(out[..., 0:2], 0.0)


def test_clear_zeroes_every_channel(ctx, bus):
    bus.composite_one(flat(ctx, (0.5, 0.5, 0.0, 0.0)), layer(destination="force"))
    bus.composite_one(flat(ctx, (0.5, 0.5, 0.0, 0.0)), layer(destination="strafe"))
    bus.clear()
    assert np.allclose(read(bus), 0.0)


def test_the_bus_resolution_scale_shrinks_the_texture(ctx):
    b = FieldBus(ctx)
    b.ensure(64, 64, 0.5)
    assert b.resolution == (32, 32)
    b.cleanup()


def test_a_scale_can_never_produce_a_zero_sized_texture(ctx):
    b = FieldBus(ctx)
    b.ensure(2, 2, 0.25)
    assert b.resolution == (1, 1)
    b.cleanup()
