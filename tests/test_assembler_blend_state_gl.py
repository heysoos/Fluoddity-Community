"""The frame assembler must not inherit a blend state from whoever ran before it.

Its pass writes the accumulation buffer with the result it computed, alpha 1.
With additive blending left on, that result is ADDED to what the buffer
already holds, so the picture grows by one every frame until bloom turns it
into NaN. The sim's trail deposit leaves blending on; the camera views happen
to switch it off in their particle pass, so only Canvas, Force Field and
Strafe Field were exposed - which is why the Canvas view went white and then
black on every click.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from state import view_modes  # noqa: E402
from utilities.frame_assembler import FrameAssembler  # noqa: E402

CAN = (32, 32)


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


def _leave_additive_blending_on(ctx):
    """What the sim's trail deposit leaves behind."""
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    ctx.blend_equation = moderngl.FUNC_ADD


@pytest.mark.parametrize("view_mode", [view_modes.CANVAS, view_modes.FORCE_FIELD,
                                       view_modes.STRAFE_FIELD, view_modes.CAMERA])
def test_a_leftover_additive_blend_does_not_accumulate(ctx, view_mode):
    src = ctx.texture(CAN, 4, dtype="f4")
    src.write(np.full((CAN[1], CAN[0], 4), 0.3, dtype="f4").tobytes())
    try:
        fa = FrameAssembler(ctx, src)
        kw = dict(view_mode=view_mode, exposure=0.0, tonemap_softness=1.52,
                  brightness=0.85, canvas_resolution=CAN, screen_aspect=1.0,
                  camera_position=(0.0, 0.0), camera_zoom=1.0)
        maxima = []
        for _ in range(30):
            _leave_additive_blending_on(ctx)
            fa.assemble_frame(src, total_samples=1, current_sample_index=0, **kw)
            a = np.frombuffer(fa.get_current_texture().read(), dtype="f4")
            assert np.isfinite(a).all()
            maxima.append(float(a.max()))
        assert maxima[-1] == pytest.approx(maxima[1], rel=1e-4), (
            f"accumulated: frame 1 {maxima[1]:.4g} -> frame 29 {maxima[-1]:.4g}")
    finally:
        ctx.disable(moderngl.BLEND)
        src.release()
