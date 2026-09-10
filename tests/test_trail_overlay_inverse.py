"""The Particles + Trails overlay inverts the transform that DREW the frame.

There are two families of view and two different transforms, and the trail
overlay was using the wrong one's inverse:

  views 0/3/4  a canvas-sized texture handed to camera.vert, which applies the
               camera at DISPLAY time. screen_to_canvas_uv inverts this, and
               the reticles that use it are correct today.
  views 1/2/5  particles RASTERISED by cam_brush.vert, which maps entity space
               by entity_to_ndc = (1/sqrt(ca), sqrt(ca)) and then by the
               letterbox fit, baking the camera into the raster.

The ratio between them is exactly 1 on a square canvas, which is why this was
invisible at 1:1 and stretched everything else.

These are pure arithmetic over the two shaders' formulas - no GL - because what
is wrong is the composition, and a GL test would only report that some pixels
moved.
"""
from __future__ import annotations

import math

import pytest

from camera import tiling_scale


def forward_cam_brush(entity_pos, canvas_size, window_size, cam_pos, cam_zoom):
    """cam_brush.vert, entity space -> screen UV. Transcribed from the shader."""
    ca = canvas_size[0] / canvas_size[1]
    entity_to_ndc = (1.0 / math.sqrt(ca), math.sqrt(ca))
    window_aspect = window_size[0] / window_size[1]
    if ca > window_aspect:
        scale = [1.0, window_aspect / ca]
    else:
        scale = [ca / window_aspect, 1.0]
    scale = [s / cam_zoom for s in scale]

    ndc = [entity_pos[0] * entity_to_ndc[0] * scale[0],
           entity_pos[1] * entity_to_ndc[1] * scale[1]]
    ndc[0] -= cam_pos[0] / cam_zoom
    ndc[1] -= -cam_pos[1] / cam_zoom
    return ((ndc[0] + 1.0) / 2.0, (ndc[1] + 1.0) / 2.0)


def inverse_new(screen_uv, canvas_size, window_size, cam_pos, cam_zoom):
    """particle_screen_to_canvas_uv, as written into frame_assembly.frag."""
    ndc = (screen_uv[0] * 2.0 - 1.0, screen_uv[1] * 2.0 - 1.0)
    ts = tiling_scale(window_size, canvas_size)
    world = [ndc[0] * cam_zoom + cam_pos[0], ndc[1] * cam_zoom + -cam_pos[1]]
    world = [world[0] * ts[0], world[1] * ts[1]]
    ca = canvas_size[0] / canvas_size[1]
    world = [world[0] * (1.0 / math.sqrt(ca)), world[1] * math.sqrt(ca)]
    return (world[0] * 0.5 + 0.5, world[1] * 0.5 + 0.5)


def inverse_old(screen_uv, canvas_size, window_size, cam_pos, cam_zoom):
    """screen_to_canvas_uv - the one the overlay used to call. Correct for
    views 0/3/4, wrong here."""
    ndc = (screen_uv[0] * 2.0 - 1.0, screen_uv[1] * 2.0 - 1.0)
    screen_aspect = window_size[0] / window_size[1]
    world = [ndc[0] * cam_zoom + cam_pos[0], ndc[1] * cam_zoom + -cam_pos[1]]
    world[0] *= screen_aspect
    return (world[0] / 2.0 + 0.5, world[1] / 2.0 + 0.5)


def entity_to_canvas_uv(entity_pos, canvas_size):
    """Where an entity-space point lives in the canvas texture."""
    ca = canvas_size[0] / canvas_size[1]
    return ((entity_pos[0] / math.sqrt(ca)) * 0.5 + 0.5,
            (entity_pos[1] * math.sqrt(ca)) * 0.5 + 0.5)


# A non-square canvas AND a non-square window, off-centre camera, zoom != 1:
# every term that could cancel is made not to. A centred camera on a square
# canvas validates any orientation bug you like.
WIDE = (1365, 768)      # 16:9-ish canvas
WINDOW = (1600, 900)
SQUARE = (1024, 1024)


@pytest.mark.parametrize("entity_pos", [
    (0.0, 0.0), (0.4, -0.3), (-0.7, 0.55), (0.9, 0.9),
])
def test_the_new_inverse_undoes_cam_brush_vert(entity_pos):
    cam_pos, cam_zoom = (0.13, -0.21), 1.7
    uv = forward_cam_brush(entity_pos, WIDE, WINDOW, cam_pos, cam_zoom)
    got = inverse_new(uv, WIDE, WINDOW, cam_pos, cam_zoom)
    want = entity_to_canvas_uv(entity_pos, WIDE)
    assert got[0] == pytest.approx(want[0], abs=1e-6)
    assert got[1] == pytest.approx(want[1], abs=1e-6)


def test_the_old_inverse_really_was_wrong_on_a_non_square_canvas():
    """Without this the test above could pass against a no-op change."""
    entity_pos, cam_pos, cam_zoom = (0.4, -0.3), (0.13, -0.21), 1.7
    uv = forward_cam_brush(entity_pos, WIDE, WINDOW, cam_pos, cam_zoom)
    want = entity_to_canvas_uv(entity_pos, WIDE)
    old = inverse_old(uv, WIDE, WINDOW, cam_pos, cam_zoom)
    assert abs(old[0] - want[0]) > 0.05 or abs(old[1] - want[1]) > 0.05, (
        "the old formula must actually differ, or this bug was elsewhere")


def test_the_two_inverses_agree_on_a_square_canvas_at_a_square_window():
    """Why nobody saw this at 1:1: the terms cancel exactly."""
    entity_pos, cam_pos, cam_zoom = (0.4, -0.3), (0.0, 0.0), 1.0
    win = (900, 900)
    uv = forward_cam_brush(entity_pos, SQUARE, win, cam_pos, cam_zoom)
    new = inverse_new(uv, SQUARE, win, cam_pos, cam_zoom)
    old = inverse_old(uv, SQUARE, win, cam_pos, cam_zoom)
    assert new[0] == pytest.approx(old[0], abs=1e-6)
    assert new[1] == pytest.approx(old[1], abs=1e-6)


def test_the_centre_of_the_canvas_lands_under_the_camera():
    """Bug 2: switching view families must not move the picture. The entity
    origin is the canvas centre, so it must map to the same screen point the
    display-time transform puts it at."""
    cam_pos, cam_zoom = (0.13, -0.21), 1.7
    uv = forward_cam_brush((0.0, 0.0), WIDE, WINDOW, cam_pos, cam_zoom)
    got = inverse_new(uv, WIDE, WINDOW, cam_pos, cam_zoom)
    assert got[0] == pytest.approx(0.5, abs=1e-6)
    assert got[1] == pytest.approx(0.5, abs=1e-6)
