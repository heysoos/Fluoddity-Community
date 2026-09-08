"""Corner pinning: the homography that squares a skewed projector to a wall.

A projector off to one side turns the rectangle it is given into a trapezoid.
The map that undoes it is a homography - a projective transform, of which
keystone correction is the special case - and four dragged corners define
exactly one.

Pure geometry: no GL, no imgui, no state. `default_corners` reaches
`fit_rect` through a deferred import so importing this module costs numpy and
nothing else.

Corners are GL coordinates over the whole projector framebuffer, v = 1 at the
TOP, ordered TL, TR, BR, BL. ImGui draws top-down, so a widget flips v.
"""
from __future__ import annotations

import numpy as np

CORNER_NAMES = ("TL", "TR", "BR", "BL")

# The source unit square, in CORNER_NAMES order.
SOURCE_CORNERS = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))

# A quad this close to folded is refused rather than handed to the shader.
_MIN_CROSS = 1e-12
_MAX_COND = 1e10


def default_corners(src_aspect: float, fb_size) -> tuple:
    """The letterbox, as corners. What an uncalibrated display starts from.

    `fit_rect` stays the one authority on where a letterboxed picture lands;
    this only normalises it.
    """
    from services.perform_window import fit_rect

    fw, fh = int(fb_size[0]), int(fb_size[1])
    if fw <= 0 or fh <= 0:
        return SOURCE_CORNERS
    x, y, w, h = fit_rect(src_aspect, (fw, fh))
    x0, x1 = x / fw, (x + w) / fw
    y0, y1 = y / fh, (y + h) / fh
    return ((x0, y1), (x1, y1), (x1, y0), (x0, y0))


def homography(dst) -> np.ndarray:
    """The 3x3 taking the source unit square to `dst`.

    Raises `numpy.linalg.LinAlgError` on a degenerate quad; callers that take
    user input want `inverse_homography`, which refuses instead.
    """
    corners = np.asarray(dst, dtype=float)
    a = np.zeros((8, 8))
    b = np.zeros(8)
    for i, ((u, v), (x, y)) in enumerate(zip(SOURCE_CORNERS, corners)):
        a[2 * i] = (u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x)
        a[2 * i + 1] = (0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y)
        b[2 * i], b[2 * i + 1] = x, y
    return np.append(np.linalg.solve(a, b), 1.0).reshape(3, 3)


def inverse_homography(dst):
    """The display-to-source map the shader samples through, or None.

    None means the quad folds, collapses or is too close to singular to
    invert. The caller keeps its last good matrix: a fold is not a picture
    anyone wants on a wall, and a NaN uniform is not a picture at all.
    """
    if not is_convex(dst):
        return None
    try:
        h = homography(dst)
        if np.linalg.cond(h) > _MAX_COND:
            return None
        inv = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None
    return inv if np.isfinite(inv).all() else None


def map_point(h, point) -> tuple:
    """Apply a 3x3 to one point, with the perspective divide."""
    q = np.asarray(h, dtype=float) @ np.array(
        [float(point[0]), float(point[1]), 1.0])
    if q[2] == 0.0:
        return (float("inf"), float("inf"))
    return (float(q[0] / q[2]), float(q[1] / q[2]))


def is_convex(corners) -> bool:
    """Whether a quad is still a quad. Winding may go either way."""
    pts = np.asarray(corners, dtype=float)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return False
    sign = 0
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross = ((b[0] - a[0]) * (c[1] - b[1])
                 - (b[1] - a[1]) * (c[0] - b[0]))
        if abs(cross) < _MIN_CROSS:
            return False                  # collinear, or an edge collapsed
        turn = 1 if cross > 0 else -1
        if sign and turn != sign:
            return False
        sign = turn
    return True


def nearest_corner(point, corners) -> int:
    """Index of the corner closest to `point`. The handle hit test."""
    pts = np.asarray(corners, dtype=float)
    d = np.sum((pts - np.asarray(point, dtype=float)) ** 2, axis=1)
    return int(np.argmin(d))


def corners_to_json(corners) -> list:
    """Four corners as preferences stores them."""
    return [[float(c[0]), float(c[1])] for c in corners]


def corners_from_json(value):
    """A stored calibration, or None when it cannot be used.

    `preferences.config` is a file a user can edit and a file an older build
    can write, so every shape reaching this is untrusted. None means "no
    calibration for this display", which is the letterbox - never a crash and
    never a fold.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            out.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            return None
    quad = tuple(out)
    return quad if is_convex(quad) else None
