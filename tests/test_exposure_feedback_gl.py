"""The exposure feedback loop must not run away on the hue-mapped views.

`frame_assembly.frag` blends the previous frame back in so exposure can work
across frames, and to do that it UNDOES the previous frame's asinh with a sinh.
That inverse is only correct for views whose stored value is the tonemapped
accumulation.

Views 0, 3 and 4 (Canvas, Force Field, Strafe Field) re-interpret the
accumulated value through `8*hsv2rgb(atan(y,x), .75, length(xy))` before the
tonemap, so the buffer holds a hue-mapped COLOUR. Applying sinh to that inverts
a function that was never applied: measured, the stored magnitude climbed about
0.84 per frame, the sinh turned that into x8.2 per frame in the value fed back,
and float32 gave out at frame 20 (exposure 0.85) or 26 (exposure 0.5). On
screen that is white spreading into black and then the picture disappearing.

Views 1, 2 and 5 never diverged, exposure 0.0 never diverged, and two or more
motion-blur samples never diverged - because at >=2 samples the compress and
the undo land in different invocations. One sample is reached by ordinary use:
motion blur off (total_samples is hardcoded to 1), a paused sim, or any
speedmult <= blur_quality.

This drives the REAL shader, because what was wrong is a property of the
recursion rather than of any one line - every source-level reading of that
branch said it was correct.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from state import view_modes  # noqa: E402
from utilities.frame_assembler import FrameAssembler  # noqa: E402

CAN = (32, 32)
FRAMES = 120

#: The defaults in state/preferences_state.py. Brightness 3.0 makes
#: BRIGHTNESS_CONSTANT 9, which is what set the old runaway's scale.
BRIGHTNESS = 3.0
SOFTNESS = 2.5

#: Views that re-interpret through hsv2rgb before the tonemap.
HUE_MAPPED = (view_modes.CANVAS, view_modes.FORCE_FIELD, view_modes.STRAFE_FIELD)
#: Views whose stored value really is the tonemapped accumulation.
TONEMAPPED = (view_modes.CAMERA, view_modes.CAMERA_TILED, view_modes.CAMERA_TRAILS)


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


def _run(ctx, view_mode, exposure, total_samples, frames=FRAMES, components=4):
    """Run the real assembler and return (frame of first non-finite or None,
    the final buffer)."""
    src = ctx.texture(CAN, components, dtype="f4")
    src.write(np.full((CAN[1], CAN[0], components), 0.3, dtype="f4").tobytes())
    try:
        fa = FrameAssembler(ctx, src)
        kw = dict(view_mode=view_mode, exposure=exposure,
                  tonemap_softness=SOFTNESS, brightness=BRIGHTNESS,
                  canvas_resolution=CAN, screen_aspect=1.0,
                  camera_position=(0.0, 0.0), camera_zoom=1.0)
        arr = None
        for f in range(frames):
            for s in range(total_samples):
                fa.assemble_frame(src, total_samples=total_samples,
                                  current_sample_index=s, **kw)
            arr = np.frombuffer(fa.get_current_texture().read(), dtype="f4")
            if not np.isfinite(arr).all():
                return f, arr
        return None, arr
    finally:
        src.release()


@pytest.mark.parametrize("view_mode", HUE_MAPPED)
@pytest.mark.parametrize("exposure", [0.25, 0.5, 0.85, 0.95, 1.0])
def test_the_hue_mapped_views_stay_finite_at_one_sample(ctx, view_mode, exposure):
    """The bug. One sample is the paused sim and the motion-blur-off path."""
    bad, arr = _run(ctx, view_mode, exposure, total_samples=1)
    assert bad is None, (
        f"view {view_mode} at exposure {exposure} went non-finite at frame {bad}")


@pytest.mark.parametrize("view_mode", HUE_MAPPED)
def test_the_hue_mapped_views_settle_rather_than_climbing(ctx, view_mode):
    """Finite is not enough - a loop with gain just under 1 would pass the test
    above and still wash the picture out. The value has to stop moving."""
    src = ctx.texture(CAN, 4, dtype="f4")
    src.write(np.full((CAN[1], CAN[0], 4), 0.3, dtype="f4").tobytes())
    try:
        fa = FrameAssembler(ctx, src)
        kw = dict(view_mode=view_mode, exposure=0.85, tonemap_softness=SOFTNESS,
                  brightness=BRIGHTNESS, canvas_resolution=CAN,
                  screen_aspect=1.0, camera_position=(0.0, 0.0), camera_zoom=1.0)
        maxima = []
        for _ in range(60):
            fa.assemble_frame(src, total_samples=1, current_sample_index=0, **kw)
            maxima.append(float(np.nanmax(
                np.frombuffer(fa.get_current_texture().read(), dtype="f4"))))
        early, late = maxima[20], maxima[-1]
        assert np.isfinite(late)
        assert abs(late - early) < 0.01 * max(1.0, abs(early)), (
            f"still drifting: frame 20 {early:.6g} -> frame 59 {late:.6g}")
    finally:
        src.release()


@pytest.mark.parametrize("view_mode", TONEMAPPED)
@pytest.mark.parametrize("exposure", [0.0, 0.5, 0.85])
def test_the_tonemapped_views_are_untouched(ctx, view_mode, exposure):
    """These never diverged and their un-tonemap is the correct inverse, so
    the guard must leave them exactly as they were."""
    bad, _ = _run(ctx, view_mode, exposure, total_samples=1)
    assert bad is None


@pytest.mark.parametrize("view_mode", HUE_MAPPED + TONEMAPPED)
@pytest.mark.parametrize("total_samples", [2, 3, 5])
def test_the_motion_blur_path_stays_finite(ctx, view_mode, total_samples):
    """At >=2 samples the compress and the undo land in different invocations,
    so this path was always safe. It must stay safe."""
    bad, _ = _run(ctx, view_mode, 0.85, total_samples=total_samples, frames=40)
    assert bad is None


@pytest.mark.parametrize("view_mode", HUE_MAPPED + TONEMAPPED)
def test_exposure_zero_is_the_shipped_default_and_must_be_stable(ctx, view_mode):
    """Exposure 0.0 never diverged - it is why this survived so long - and
    every preset in the library was tuned at it."""
    bad, arr = _run(ctx, view_mode, 0.0, total_samples=1)
    assert bad is None
    assert np.isfinite(arr).all()
