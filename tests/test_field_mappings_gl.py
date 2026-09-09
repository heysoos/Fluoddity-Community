"""What each mapping does to a picture, run through the real composite.

A mapping is a branch of one shader, so a source-level reading of it proves
nothing about which branch the index selects - the index is `MAPPINGS.index`,
and the shader's numbering has to agree with the tuple's order.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.field_sources import FrameContext  # noqa: E402
from state.field_stack import FieldLayer, FieldStack  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 64
FRAME = FrameContext(time=0.0, frame_count=0, mouse=(0, 0), prev_mouse=(0, 0),
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


class _FixedSource:
    error = None
    animated = False

    def __init__(self, tex):
        self._tex = tex

    def evaluate(self, bus, layer, frame):
        return self._tex

    def release(self):
        pass


def _texture(ctx, rgb):
    """rgb: (h, w, 3) float in 0..1."""
    data = np.ascontiguousarray(rgb, dtype="f4")
    h, w, _ = data.shape
    return ctx.texture((w, h), 3, data.tobytes(), dtype="f4")


def _render(bus, tex, mapping, **kw):
    layer = FieldLayer(source="image", mapping=mapping, blend="replace", **kw)
    bus.ensure(RES, RES, 1.0)
    bus._sources[layer.uid] = ("image", _FixedSource(tex))
    bus.mark_dirty()
    bus.rebuild(FieldStack(layers=[layer]), RES, RES, 1.0, FRAME)
    return np.frombuffer(bus.field_texture.read(),
                         dtype="f4").reshape(RES, RES, 4)[..., :2]


def _flat(ctx, r, g):
    return _texture(ctx, np.dstack([np.full((RES, RES), v, dtype=np.float32)
                                    for v in (r, g, 0.0)]))


def _bar(ctx):
    """A white bar down the middle of a black frame: two vertical edges."""
    img = np.zeros((RES, RES, 3), dtype=np.float32)
    img[:, RES // 4: 3 * RES // 4, :] = 1.0
    return _texture(ctx, img)


def test_direct_rg_pushes_one_way_on_a_positive_picture(ctx, bus):
    """The complaint: a camera's channels never go negative."""
    v = _render(bus, _flat(ctx, 0.6, 0.7), "rg_direct")
    assert v[..., 0].min() > 0.0 and v[..., 1].min() > 0.0


def test_centred_rg_puts_mid_grey_at_rest(ctx, bus):
    v = _render(bus, _flat(ctx, 0.5, 0.5), "rg_signed")
    assert abs(float(np.abs(v).max())) < 1e-5, "mid grey still pushed"


def test_centred_rg_can_push_either_way(ctx, bus):
    dark = _render(bus, _flat(ctx, 0.1, 0.1), "rg_signed")[..., 0].mean()
    light = _render(bus, _flat(ctx, 0.9, 0.9), "rg_signed")[..., 0].mean()
    assert dark < 0.0 < light, f"both ends came out the same sign: {dark} {light}"


def test_edges_ignore_a_flat_wall(ctx, bus):
    """Bright but featureless: an edge mapping must find nothing in it."""
    v = _render(bus, _flat(ctx, 0.9, 0.9), "edge")
    assert float(np.abs(v).max()) < 1e-4, "a flat picture produced an edge"


def test_edges_fire_at_the_boundary_and_nowhere_else(ctx, bus):
    v = _render(bus, _bar(ctx), "edge")
    strength = np.hypot(v[..., 0], v[..., 1])
    row = strength[RES // 2]
    edge = max(row[RES // 4 - 1: RES // 4 + 2].max(),
               row[3 * RES // 4 - 2: 3 * RES // 4 + 1].max())
    middle = row[RES // 2 - 4: RES // 2 + 4].max()
    assert edge > 0.1, f"the edge scored {edge}"
    assert middle < 0.05, f"the flat middle scored {middle}"


def test_edge_flow_runs_along_the_edge_that_edge_crosses(ctx, bus):
    """The two are a quarter turn apart, which is the whole difference."""
    across = _render(bus, _bar(ctx), "edge")
    along = _render(bus, _bar(ctx), "edge_flow")
    strength = np.hypot(across[..., 0], across[..., 1])
    hot = strength > 0.5 * strength.max()
    assert hot.any(), "nothing lit up"
    dots = (across[..., 0] * along[..., 0] + across[..., 1] * along[..., 1])[hot]
    assert float(np.abs(dots).max()) < 1e-3, "the two mappings point the same way"


def test_the_direction_control_reverses_an_edge(ctx, bus):
    out = _render(bus, _bar(ctx), "edge", sign=1.0)
    back = _render(bus, _bar(ctx), "edge", sign=-1.0)
    assert np.allclose(out, -back, atol=1e-5)


def test_the_edge_heading_does_not_depend_on_contrast(ctx, bus):
    """Direction and strength are kept apart: only the length follows the
    contrast, and the heading is the same on a faint picture as a bold one."""
    def at_the_edge(img):
        v = _render(bus, _texture(ctx, img), "edge")
        strength = np.hypot(v[..., 0], v[..., 1])
        where = np.unravel_index(strength.argmax(), strength.shape)
        peak = float(strength[where])
        return v[where] / peak, peak

    faint = np.zeros((RES, RES, 3), dtype=np.float32)
    faint[:, RES // 2:, :] = 0.05
    bold = np.zeros((RES, RES, 3), dtype=np.float32)
    bold[:, RES // 2:, :] = 1.0

    faint_dir, faint_len = at_the_edge(faint)
    bold_dir, bold_len = at_the_edge(bold)
    assert np.allclose(faint_dir, bold_dir, atol=1e-3), (
        f"the heading moved with contrast: {faint_dir} against {bold_dir}")
    assert faint_len < bold_len, "contrast reached the magnitude as nothing"


def test_edge_strength_does_not_run_away(ctx, bus):
    """Normalised direction times a clamped magnitude: never past 1."""
    v = _render(bus, _bar(ctx), "edge")
    assert float(np.hypot(v[..., 0], v[..., 1]).max()) <= 1.0 + 1e-4


def _step(ctx):
    """Black on the left, white on the right: an asymmetric picture, so a
    mirror that did nothing cannot pass."""
    img = np.zeros((RES, RES, 3), dtype=np.float32)
    img[:, RES // 2:, :] = 1.0
    return _texture(ctx, img)


def test_flip_x_mirrors_the_picture(ctx, bus):
    out = _render(bus, _step(ctx), "luminance")
    flipped = _render(bus, _step(ctx), "luminance", flip_x=True)
    assert not np.allclose(out, flipped), "the flip did nothing"
    assert np.allclose(out[..., 0], flipped[:, ::-1, 0], atol=1e-5)


def test_flip_x_mirrors_a_directly_read_vector(ctx, bus):
    """A derivative mapping picks the mirror up from its sample offsets; a
    direct read has to be turned round explicitly."""
    v = _render(bus, _flat(ctx, 0.9, 0.6), "rg_signed", flip_x=True)
    plain = _render(bus, _flat(ctx, 0.9, 0.6), "rg_signed")
    assert np.allclose(v[..., 0], -plain[..., 0], atol=1e-5)
    assert np.allclose(v[..., 1], plain[..., 1], atol=1e-5)


def test_flip_x_reverses_a_gradient(ctx, bus):
    out = _render(bus, _step(ctx), "gradient")
    flipped = _render(bus, _step(ctx), "gradient", flip_x=True)
    assert np.allclose(out[..., 0], -flipped[:, ::-1, 0], atol=1e-5)
    assert np.allclose(out[..., 1], flipped[:, ::-1, 1], atol=1e-5)
