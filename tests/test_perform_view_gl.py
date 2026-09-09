"""The projector's frame is FIXED, on a real GPU.

The camera is baked into cam_brush_target during rasterisation, so no
display-time transform can undo a zoom - the projector has to redo the particle
pass with an identity camera. That is a property of what the SHADER draws, so a
source-level reading cannot see it: these tests render and compare pixels.

Skipped when no GL context is available (CI, remote shells, no GPU).
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")
glfw = pytest.importorskip("glfw")

PROJECTOR = (320, 180)
LAPTOP = (256, 256)


@pytest.fixture(scope="module")
def rig():
    """A hidden window, a sim, a camera and a PerformView over them.

    A hidden WINDOW rather than a standalone context: Camera calls
    glfw.get_framebuffer_size(self.window) in several places, and borrowing the
    real Camera is what keeps this from drifting away from what the app draws.
    """
    if not glfw.init():
        pytest.skip("glfw would not initialise")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(*LAPTOP, "perform-view-test", None, None)
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
    from services.perform_view import PerformView

    # A small world: this test is about WHERE things are drawn, not how many.
    sim = Sim(ctx, world_size=0.05, canvas_aspect_ratio="1:1",
              particle_density=1.0)
    camera = Camera(ctx, sim, window)
    camera.cam_brush_mode = True
    view = PerformView(ctx, sim, camera)
    # A few steps so the canvas is not empty.
    for _ in range(4):
        sim.update(ctx)
    yield ctx, sim, camera, view
    view.cleanup()
    glfw.terminate()


def _state():
    from state.ui_state import UIState
    from state import view_modes

    s = UIState()
    s.sim.current_view_option = view_modes.CAMERA
    s.preferences.bloom_enabled = False      # one variable at a time
    return s


def _kwargs(camera, sim, screen_aspect=1.0):
    from state import view_modes

    return dict(
        view_mode=view_modes.CAMERA, sweep_mode=False,
        sweep_reticle_pos=(0.5, 0.5), sweep_reticle_visible=False,
        screen_aspect=screen_aspect, brightness=camera.BRIGHTNESS,
        exposure=0.0, ink_weight=1.0, watercolor_mode=False,
        camera_position=tuple(camera.position), camera_zoom=camera.zoom,
        trail_draw_radius=0.0, mouse_screen_coords=(0.5, 0.5),
        tiling_mode=False, view_min=(0.0, 0.0), view_max=(0.0, 0.0),
        tiling_scale=camera.compute_tiling_scale(),
        canvas_resolution=sim.get_canvas_dimensions(), tonemap_softness=1.0,
    )


def _perform_pixels(rig, cam_pos, cam_zoom):
    """Render the projector's frame with the LAPTOP camera set as given."""
    ctx, sim, camera, view = rig
    camera.position = np.array(list(cam_pos), dtype=float)
    camera.zoom = cam_zoom
    st = _state()
    view.render(st, _kwargs(camera, sim), PROJECTOR, 1, 0)
    assert view.frame is not None, "no frame published on the final sample"
    tex = view.frame.texture
    raw = tex.read()
    return np.frombuffer(raw, dtype="f4").reshape(tex.size[1], tex.size[0], -1)


def _laptop_pixels(rig, cam_pos, cam_zoom):
    """Render the LAPTOP's own view with the same camera, for contrast."""
    ctx, sim, camera, view = rig
    camera.position = np.array(list(cam_pos), dtype=float)
    camera.zoom = cam_zoom
    raw_tex = camera.generate_view_texture(tiling_mode=False)
    out = camera.frame_assembler.assemble_frame(
        raw_tex, total_samples=1, current_sample_index=0,
        **_kwargs(camera, sim))
    data = np.frombuffer(out.read(), dtype="f4")
    return data.reshape(out.size[1], out.size[0], -1)


def _race_floor(rig):
    """How much two renders of ONE scene differ from each other.

    Particles splat additively into a shared texture, which races, so the
    projector's frame is not bit-reproducible and array_equal is the wrong
    instrument - it fails on a correct build. Everything below is measured
    against this floor instead, which is what makes "the camera did not reach
    it" a real assertion rather than a tolerance someone guessed.
    """
    a = _perform_pixels(rig, (0.0, 0.0), 1.0)
    b = _perform_pixels(rig, (0.0, 0.0), 1.0)
    return float(np.abs(a - b).max())


def test_the_projector_ignores_the_laptop_zoom(rig):
    """The whole point: zooming the laptop must not move the projection."""
    floor = _race_floor(rig)
    a = _perform_pixels(rig, (0.0, 0.0), 1.0)
    b = _perform_pixels(rig, (0.0, 0.0), 4.0)
    moved = float(np.abs(a - b).max())
    assert moved <= max(floor, 1e-3), (
        f"the laptop's zoom reached the projector: moved {moved:.3e} "
        f"against a splat-race floor of {floor:.3e}")


def test_the_projector_ignores_the_laptop_pan(rig):
    floor = _race_floor(rig)
    a = _perform_pixels(rig, (0.0, 0.0), 1.0)
    b = _perform_pixels(rig, (0.6, -0.4), 1.0)
    moved = float(np.abs(a - b).max())
    assert moved <= max(floor, 1e-3), (
        f"the laptop's pan reached the projector: moved {moved:.3e} "
        f"against a splat-race floor of {floor:.3e}")


def test_the_race_floor_is_far_below_a_real_camera_move(rig):
    """Otherwise the two tests above pass on any tolerance at all.

    The laptop's own view is the control: moving its camera changes its
    picture by orders of magnitude more than the splat race does.
    """
    floor = _race_floor(rig)
    a = _laptop_pixels(rig, (0.0, 0.0), 1.0)
    b = _laptop_pixels(rig, (0.0, 0.0), 4.0)
    real = float(np.abs(a - b).max())
    assert real > floor * 100, (
        f"a real camera move is only {real:.3e} against a floor of "
        f"{floor:.3e}; the comparison cannot tell them apart")


def test_the_laptop_view_really_does_change(rig):
    """Otherwise the two tests above pass because NOTHING is being drawn."""
    a = _laptop_pixels(rig, (0.0, 0.0), 1.0)
    b = _laptop_pixels(rig, (0.0, 0.0), 4.0)
    assert not np.array_equal(a, b), "the camera moved and the view did not"


def test_the_projector_frame_is_the_projectors_shape(rig):
    _perform_pixels(rig, (0.0, 0.0), 1.0)
    _, _, _, view = rig
    assert view.frame.texture.size == PROJECTOR
    # An identity DisplayFrame: the perform window's letterbox is a no-op,
    # because this texture is already at the display's shape.
    assert view.frame.cam_pos == (0.0, 0.0)
    assert view.frame.cam_zoom == 1.0
    assert view.frame.window_size == PROJECTOR


def test_the_projector_draws_something(rig):
    """A fixed viewpoint that renders black would pass every test above."""
    img = _perform_pixels(rig, (0.0, 0.0), 1.0)
    assert img[..., :3].any(), "the projector's frame is entirely black"


def test_the_whole_canvas_is_inside_the_frame(rig):
    """At zoom 1 the canvas is FITTED, so the outer columns are background.

    The canvas is square and the projector is 16:9, so a frame with content
    running to the left and right edges means the canvas was cropped rather
    than fitted.
    """
    img = _perform_pixels(rig, (0.0, 0.0), 1.0)
    h, w, _ = img.shape
    margin = (w - h) // 2
    assert margin > 0
    edge = max(img[:, :margin // 2, :3].max(), img[:, -(margin // 2):, :3].max())
    assert edge == 0.0, "content reaches the side edges; the canvas is cropped"


def test_a_resize_rebuilds_and_republishes(rig):
    """A monitor change moves the target; a stale one is the wrong shape."""
    _, sim, camera, view = rig
    _perform_pixels(rig, (0.0, 0.0), 1.0)
    st = _state()
    view.render(st, _kwargs(camera, sim), (200, 200), 1, 0)
    assert view.frame.texture.size == (200, 200)


def test_accumulating_publishes_only_on_the_final_sample(rig):
    """Motion blur: the projector must not show a partial accumulation."""
    _, sim, camera, view = rig
    view.resize(PROJECTOR)
    view.frame = None
    st = _state()
    view.render(st, _kwargs(camera, sim), PROJECTOR, 3, 0)
    assert view.frame is None
    view.render(st, _kwargs(camera, sim), PROJECTOR, 3, 1)
    assert view.frame is None
    view.render(st, _kwargs(camera, sim), PROJECTOR, 3, 2)
    assert view.frame is not None


def test_it_puts_back_the_target_it_found(rig):
    """Paused, this is the last pass before imgui - so a target left bound is
    the one the UI draws into, and that buffer is what the projector shows:
    the panels leave the laptop and appear on the wall.

    A scratch framebuffer stands in for the screen, which a hidden window has
    but a standalone context does not.
    """
    ctx, sim, camera, view = rig
    scratch = ctx.framebuffer(color_attachments=[ctx.texture((64, 64), 4)])
    try:
        scratch.use()
        view.render(_state(), _kwargs(camera, sim), PROJECTOR, 1, 0)
        assert ctx.fbo is scratch, "the projector's own buffer was left bound"
    finally:
        scratch.release()


def test_it_puts_the_target_back_in_the_other_view_modes(rig):
    """The canvas views take an early return out of the particle pass, so they
    reach the assembler by another road and leak a different buffer."""
    ctx, sim, camera, view = rig
    scratch = ctx.framebuffer(color_attachments=[ctx.texture((64, 64), 4)])
    was = camera.cam_brush_mode
    try:
        scratch.use()
        camera.cam_brush_mode = False
        view.render(_state(), _kwargs(camera, sim), PROJECTOR, 1, 0)
        assert ctx.fbo is scratch, "the projector's own buffer was left bound"
    finally:
        camera.cam_brush_mode = was
        scratch.release()
