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
    ctx, sim, camera, view = rig
    tex = view.render(_state(), _kwargs(camera, sim), SIDE,
                      grayscale=grayscale)
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
