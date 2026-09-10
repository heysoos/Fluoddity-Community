"""Wrapping the camera back into one canvas period.

The Tiled view repeats the canvas across the plane, so a pan there is unbounded.
Every other view shows exactly one canvas, so leaving Tiled folds the camera
back into a single period.

The wrap must be the IDENTITY inside that period. The formula this replaces was
`fmod(p + 100, 2) - 1`, where 100 is a multiple of the period - so the offset
shifted nothing, the `-1` was left over, and every position moved by a half
period, 0.0 included. It was also not idempotent, so repeated view switches
walked the camera across the world.
"""
from __future__ import annotations

#: The canvas occupies [-1, 1) in camera-position units.
PERIOD = 2.0


def wrap_camera_position(p: float) -> float:
    """Fold `p` into [-1, 1), leaving anything already there untouched."""
    return (float(p) + 1.0) % PERIOD - 1.0
