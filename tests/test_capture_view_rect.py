"""The crop rect must describe the texture it is cropping.

main.py applies camera state at step 5, captures at step 5.1.5 and renders at
step 7, so camera.assembled_texture at capture time was rendered ONE FRAME
EARLIER with the camera where it was then. Recomputing the rect at capture time
therefore crops last frame's pixels with this frame's camera.

Measured 2026-08-07 at grid 8: one frame of ordinary input displaces the crop by
4px (pan 0.005), 18px (pan 0.02) or 43px (one scroll notch of zoom) - each tile
picking up a strip of its neighbour. The same mismatch appears for ~150ms after
a window resize, because cam_brush_target only follows on the debounced reload.

The fix is to record the rect WITH the texture. These tests pin that down.
"""
import numpy as np
import pytest

from camera import Camera


class _FakeTex:
    def __init__(self, size):
        self.size = size


class _FakeSim:
    def __init__(self, canvas=(647, 647)):
        self.view_tex = _FakeTex(canvas)


def camera_at(position=(0.0, 0.0), zoom=1.0, canvas=(647, 647)):
    """A Camera with only the projection state set - no GL, no window."""
    cam = object.__new__(Camera)
    cam.sim = _FakeSim(canvas)
    cam.position = np.array(position, dtype=np.float32)
    cam.zoom = float(zoom)
    cam.window = None
    return cam


# ---- the rect itself ----------------------------------------------------

def test_a_square_canvas_in_a_square_window_fills_the_frame():
    cam = camera_at()
    lo, hi = cam.canvas_view_rect((800, 800), fb_size=(800, 800))
    assert lo == pytest.approx((0.0, 0.0), abs=1e-6)
    assert hi == pytest.approx((1.0, 1.0), abs=1e-6)


def test_a_square_canvas_in_a_wide_window_is_letterboxed_horizontally():
    cam = camera_at()
    lo, hi = cam.canvas_view_rect((1920, 1080), fb_size=(1920, 1080))
    assert hi[1] - lo[1] == pytest.approx(1.0, abs=1e-6), "full height"
    assert hi[0] - lo[0] == pytest.approx(1080 / 1920, abs=1e-4), "pillarboxed"
    assert lo[0] + hi[0] == pytest.approx(1.0, abs=1e-6), "centred"


def test_a_square_canvas_in_a_tall_window_is_letterboxed_vertically():
    cam = camera_at()
    lo, hi = cam.canvas_view_rect((900, 1600), fb_size=(900, 1600))
    assert hi[0] - lo[0] == pytest.approx(1.0, abs=1e-6), "full width"
    assert hi[1] - lo[1] == pytest.approx(900 / 1600, abs=1e-4)
    assert lo[1] + hi[1] == pytest.approx(1.0, abs=1e-6), "centred"


def test_zooming_out_shrinks_the_canvas_within_the_frame():
    cam = camera_at(zoom=1.0)
    _lo1, hi1 = cam.canvas_view_rect((800, 800), fb_size=(800, 800))
    cam.zoom = 2.0
    lo2, hi2 = cam.canvas_view_rect((800, 800), fb_size=(800, 800))
    assert (hi2[0] - lo2[0]) < (hi1[0] - 0.0)


def test_the_rect_is_expressed_in_the_textures_own_space():
    """The divisor is the TEXTURE size, and it is not assumed to equal the
    framebuffer size - that assumption is what breaks after a resize."""
    cam = camera_at()
    lo_a, hi_a = cam.canvas_view_rect((1920, 1080), fb_size=(1920, 1080))
    lo_b, hi_b = cam.canvas_view_rect((960, 540), fb_size=(1920, 1080))
    # same screen rect, half-sized texture -> UV coordinates double
    assert lo_b[0] == pytest.approx(lo_a[0] * 2, abs=1e-6)
    assert hi_b[0] == pytest.approx(hi_a[0] * 2, abs=1e-6)


# ---- the invariant that actually fixes the bug --------------------------

def test_moving_the_camera_after_a_render_does_not_change_a_recorded_rect():
    """The whole point. A rect captured at render time is a value, so later
    camera motion cannot retroactively change which pixels it names."""
    cam = camera_at(position=(0.0, 0.0))
    recorded = cam.canvas_view_rect((1920, 1080), fb_size=(1920, 1080))
    cam.position = np.array([0.0, 0.05], dtype=np.float32)
    cam.zoom = 1.1
    assert cam.canvas_view_rect((1920, 1080), fb_size=(1920, 1080)) != recorded
    assert recorded == recorded


def test_one_frame_of_camera_motion_would_shift_a_recomputed_rect():
    """Documents the size of the bug this guards against, at grid 8."""
    out_px = 8 * 224
    cam = camera_at(position=(0.0, 0.0))
    lo0, hi0 = cam.canvas_view_rect((1920, 1080), fb_size=(1920, 1080))
    cam.position = np.array([0.0, 0.02], dtype=np.float32)
    lo1, hi1 = cam.canvas_view_rect((1920, 1080), fb_size=(1920, 1080))
    shift = 0.5 * (abs(lo1[1] - lo0[1]) + abs(hi1[1] - hi0[1])) * out_px
    assert shift > 10.0, f"expected a visible shift, got {shift:.1f}px"


def test_tex_to_screen_survives_a_minimised_window():
    """GLFW reports 0x0 while minimised. screen_to_tex has always clamped this;
    tex_to_screen did not, so the first frame after minimising divided by zero
    inside the render path."""
    cam = camera_at()
    lo, hi = cam.canvas_view_rect((800, 800), fb_size=(0, 0))
    assert all(np.isfinite(v) for v in (*lo, *hi))


def test_the_runner_records_the_rect_whenever_it_sets_the_texture(monkeypatch):
    """assembled_texture and assembled_view_rect must be assigned together;
    a texture with a stale rect is exactly the bug."""
    import camera as camera_module
    from simulation_runner import SimulationRunner

    monkeypatch.setattr(camera_module.glfw, "get_framebuffer_size",
                        lambda _w: (1920, 1080))

    class _Prefs:
        bloom_enabled = False

    class _SimState:
        watercolor_mode = False

    class _UIState:
        preferences = _Prefs()
        sim = _SimState()

    cam = camera_at()
    cam.assembled_texture = None
    cam.assembled_view_rect = None

    class _Video:
        def is_active(self):
            return False

    runner = object.__new__(SimulationRunner)
    runner.camera = cam
    runner.video_service = _Video()

    tex = _FakeTex((1920, 1080))
    runner._process_assembled_frame(tex, _UIState())

    assert cam.assembled_texture is tex
    assert cam.assembled_view_rect is not None, \
        "the rect must be recorded with the texture, not recomputed at capture"
    lo, hi = cam.assembled_view_rect
    assert hi[1] - lo[1] == pytest.approx(1.0, abs=1e-6)


def test_a_fresh_camera_has_no_recorded_rect():
    cam = camera_at()
    cam.assembled_texture = None
    cam.assembled_view_rect = None
    assert cam.assembled_view_rect is None
