"""A layer may read either channel pair of its source.

The legacy field texture packed force in .xy and strafe in .zw of one image.
Both migrated layers read one brush buffer, so without a source swizzle the
strafe half of every pre-existing preset silently becomes a copy of its force
half.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from state.field_stack import FieldLayer  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 16


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


def packed(ctx):
    """force=(0.1,0.2) in .xy, strafe=(0.7,0.8) in .zw - the legacy layout."""
    data = np.tile(np.array([0.1, 0.2, 0.7, 0.8], dtype="f4"), (RES, RES, 1))
    return ctx.texture((RES, RES), 4, data.tobytes(), dtype="f4")


def read(bus):
    return np.frombuffer(bus.field_texture.read(), dtype="f4").reshape(RES, RES, 4)


def test_the_default_reads_xy(ctx, bus):
    bus.clear()
    bus.composite_one(packed(ctx), FieldLayer(
        mapping="rg_direct", destination="force", blend="replace"))
    out = read(bus)
    assert np.allclose(out[..., 0], 0.1)
    assert np.allclose(out[..., 1], 0.2)


def test_zw_reads_the_second_pair(ctx, bus):
    bus.clear()
    bus.composite_one(packed(ctx), FieldLayer(
        mapping="rg_direct", destination="strafe", blend="replace",
        params={"_channels": "zw"}))
    out = read(bus)
    assert np.allclose(out[..., 2], 0.7)
    assert np.allclose(out[..., 3], 0.8)


def test_the_legacy_pair_reconstructs_the_old_field(ctx, bus):
    src = packed(ctx)
    bus.clear()
    bus.composite_one(src, FieldLayer(
        mapping="rg_direct", destination="force", blend="replace",
        params={"_channels": "xy"}))
    bus.composite_one(src, FieldLayer(
        mapping="rg_direct", destination="strafe", blend="replace",
        params={"_channels": "zw"}))
    assert np.allclose(read(bus)[0, 0], [0.1, 0.2, 0.7, 0.8])
