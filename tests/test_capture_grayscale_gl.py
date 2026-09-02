"""Greyscale scoring renders DENSITY, on a real GPU.

The capture zeroes every particle's saturation in the particle pass, so what
the encoder sees carries no colour term at all. That is a property of what the
SHADER draws, and the program is shared with the laptop's own view, so these
tests render and read pixels: the capture must be grey, the ordinary capture
must not be, and the view drawn AFTER a grey capture must still be in colour.

Skipped when no GL context is available (CI, remote shells, no GPU).
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")
glfw = pytest.importorskip("glfw")

SIDE = 224
WINDOW = (256, 256)


@pytest.fixture(scope="module")
def rig():
    if not glfw.init():
        pytest.skip("glfw would not initialise")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(*WINDOW, "capture-grey-test", None, None)
    if not window:
        glfw.terminate()
        pytest.skip("no window, no GL")
    glfw.make_context_current(window)
    try:
        ctx = moderngl.create_context()
    except Exception as exc:
        glfw.terminate()
        pytest.skip(f"no GL context: {exc}")

    from sim import Sim
    from camera import Camera
    from services.capture_view import CaptureView

    sim = Sim(ctx, world_size=0.05, canvas_aspect_ratio="1:1",
              particle_density=1.0)
    camera = Camera(ctx, sim, window)
    camera.cam_brush_mode = True
    view = CaptureView(ctx, sim, camera)
    for _ in range(4):
        sim.update(ctx)
    yield ctx, sim, camera, view
    view.release()
    glfw.terminate()


def _state():
    from state import view_modes
    from state.ui_state import UIState

    s = UIState()
    s.sim.current_view_option = view_modes.CAMERA
    s.preferences.bloom_enabled = False
    return s


def _kwargs(camera, sim):
    from state import view_modes

    return dict(
        view_mode=view_modes.CAMERA, sweep_mode=False,
        sweep_reticle_pos=(0.5, 0.5), sweep_reticle_visible=False,
        screen_aspect=1.0, brightness=camera.BRIGHTNESS,
        exposure=0.0, ink_weight=1.0, watercolor_mode=False,
        camera_position=tuple(camera.position), camera_zoom=camera.zoom,
        trail_draw_radius=0.0, mouse_screen_coords=(0.5, 0.5),
        tiling_mode=False, view_min=(0.0, 0.0), view_max=(0.0, 0.0),
        tiling_scale=camera.compute_tiling_scale(),
        canvas_resolution=sim.get_canvas_dimensions(), tonemap_softness=1.0,
    )


def _pixels(tex):
    data = np.frombuffer(tex.read(), dtype="f4")
    return data.reshape(tex.size[1], tex.size[0], -1)[..., :3]


def _capture(rig, grayscale):
    return _capture_in(rig, None, grayscale)


def _render_twice(view, kw, grayscale):
    """The assembler's first-frame mix keeps a 1e-4 sliver of the PREVIOUS
    accumulation buffer, so the second of two identical renders is the one
    that says what this render draws."""
    for _ in range(2):
        tex = view.render(_state(), kw, SIDE, grayscale=grayscale)
    assert tex is not None
    return _pixels(tex)


def _chroma(px):
    """How far the channels disagree anywhere in the frame."""
    return float(max(np.abs(px[..., 0] - px[..., 1]).max(),
                     np.abs(px[..., 1] - px[..., 2]).max()))


def test_a_grey_capture_draws_something_with_no_colour(rig):
    px = _capture(rig, True)
    assert px.max() > 0.0, "nothing was drawn"
    assert _chroma(px) < 1e-4


def test_an_ordinary_capture_is_still_in_colour(rig):
    assert _chroma(_capture(rig, False)) > 1e-3


def test_the_view_after_a_grey_capture_is_still_in_colour(rig):
    """The uniform lives on the program the laptop's own view draws with."""
    ctx, sim, camera, view = rig
    _capture(rig, True)
    raw = camera.generate_view_texture(tiling_mode=False)
    assert _chroma(_pixels(raw)) > 1e-3


# -- the OTHER two colour sources ------------------------------------------
# The capture follows the live view mode. The Canvas view and the trail
# overlay under Camera + Trails both paint flow DIRECTION as hue, in the
# assembler rather than the particle pass, so zeroing saturation there alone
# left the encoder looking at colour whenever the view was not plain Camera.

def _capture_in(rig, view_mode, grayscale, **extra):
    from state import view_modes

    ctx, sim, camera, view = rig
    if view_mode is None:
        view_mode = view_modes.CAMERA
    was = camera.cam_brush_mode
    camera.cam_brush_mode = view_mode in view_modes.CAMERA_VIEWS
    try:
        kw = _kwargs(camera, sim)
        kw["view_mode"] = view_mode
        kw.update(extra)
        return _render_twice(view, kw, grayscale)
    finally:
        camera.cam_brush_mode = was


def test_the_canvas_view_is_grey_when_asked(rig):
    from state import view_modes

    colour = _capture_in(rig, view_modes.CANVAS, False)
    assert _chroma(colour) > 1e-3, "the canvas view paints direction as hue"
    grey = _capture_in(rig, view_modes.CANVAS, True)
    assert grey.max() > 0.0
    assert _chroma(grey) < 1e-4


def test_the_trail_overlay_is_grey_when_asked(rig):
    from state import view_modes

    ctx, sim, camera, view = rig
    overlay = dict(trail_tex=sim.view_tex, trail_overlay_strength=1.0)
    colour = _capture_in(rig, view_modes.CAMERA_TRAILS, False, **overlay)
    assert _chroma(colour) > 1e-3, "the overlay paints direction as hue"
    grey = _capture_in(rig, view_modes.CAMERA_TRAILS, True, **overlay)
    assert grey.max() > 0.0
    assert _chroma(grey) < 1e-4
