"""The corner-pin homography: pure geometry, no GL.

Every quad here is ASYMMETRIC on purpose. A symmetric one cancels the v flip
and passes under either orientation convention, which is the same trap the
capture-crop tests avoid by panning the camera.
"""
import numpy as np
import pytest

from services.corner_pin import (
    CORNER_NAMES,
    default_corners,
    homography,
    inverse_homography,
    is_convex,
    map_point,
    nearest_corner,
)
from services.perform_window import fit_rect


# An off-centre trapezoid: no two edges the same length, no mirror symmetry.
SKEW = ((0.10, 0.95), (0.90, 0.80), (0.97, 0.12), (0.05, 0.05))
FULL = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))


# --- naming and orientation ----------------------------------------------

def test_the_corners_are_named_clockwise_from_the_top_left():
    assert CORNER_NAMES == ("TL", "TR", "BR", "BL")


def test_v_is_one_at_the_top():
    """GL coordinates: the TL corner sits at the HIGH end of v, not the low."""
    tl, tr, br, bl = default_corners(1.0, (1000, 1000))
    assert tl[1] > bl[1]
    assert tr[1] > br[1]


def test_each_source_corner_lands_on_its_own_destination_corner():
    """The ordering contract. A rotated or flipped quad fails here first."""
    h = homography(SKEW)
    src = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
    for name, s, d in zip(CORNER_NAMES, src, SKEW):
        got = map_point(h, s)
        assert got == pytest.approx(d, abs=1e-9), f"{name} landed at {got}"


def test_raising_the_top_left_corner_moves_the_top_of_the_display():
    """The orientation pin. Nothing below the middle may move."""
    corners = list(FULL)
    corners[0] = (0.30, 1.0)          # pull TL inward along the top edge
    h = homography(tuple(corners))

    # The source's top-left goes with it; the source's bottom-left does not.
    assert map_point(h, (0.0, 1.0))[0] == pytest.approx(0.30, abs=1e-9)
    assert map_point(h, (0.0, 0.0))[0] == pytest.approx(0.0, abs=1e-9)


# --- default corners ------------------------------------------------------

def test_a_matching_aspect_defaults_to_the_whole_display():
    assert default_corners(1920 / 1080, (1920, 1080)) == pytest.approx(
        np.array(FULL))


def test_the_defaults_are_the_letterbox():
    """One source of truth: fit_rect decides, default_corners normalises it."""
    for src in (0.5, 1.0, 2.0, 2.35):
        for fb in ((1920, 1080), (1024, 768), (1080, 1920)):
            x, y, w, h = fit_rect(src, fb)
            got = default_corners(src, fb)
            xs = {round(c[0], 9) for c in got}
            ys = {round(c[1], 9) for c in got}
            assert xs == {round(x / fb[0], 9), round((x + w) / fb[0], 9)}
            assert ys == {round(y / fb[1], 9), round((y + h) / fb[1], 9)}


def test_a_wider_source_defaults_to_bars_top_and_bottom():
    tl, tr, br, bl = default_corners(2.0, (1000, 1000))
    assert (tl[0], tr[0]) == (0.0, 1.0)
    assert tl[1] == pytest.approx(0.75)
    assert bl[1] == pytest.approx(0.25)


# --- the map itself -------------------------------------------------------

def test_the_full_display_quad_is_the_identity():
    h = homography(FULL)
    for p in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.25, 0.8)):
        assert map_point(h, p) == pytest.approx(p, abs=1e-9)


def test_the_centre_lands_where_the_diagonals_cross():
    """A projective invariant, so it is hand-checkable under any warp."""
    h = homography(SKEW)
    tl, tr, br, bl = (np.array(c, dtype=float) for c in SKEW)

    # Intersect TL->BR with TR->BL.
    d1, d2 = br - tl, bl - tr
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    t = ((tr[0] - tl[0]) * d2[1] - (tr[1] - tl[1]) * d2[0]) / denom
    expected = tl + t * d1

    assert map_point(h, (0.5, 0.5)) == pytest.approx(expected, abs=1e-9)


def test_a_straight_line_stays_straight():
    """What makes the warped calibration grid worth drawing."""
    h = homography(SKEW)
    pts = np.array([map_point(h, (t, 0.3)) for t in np.linspace(0, 1, 9)])
    a, b = pts[0], pts[-1]
    for p in pts[1:-1]:
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        assert cross == pytest.approx(0.0, abs=1e-9)


def test_the_inverse_undoes_the_forward_map():
    h = homography(SKEW)
    inv = inverse_homography(SKEW)
    for p in ((0.0, 0.0), (0.13, 0.77), (0.5, 0.5), (1.0, 1.0)):
        assert map_point(inv, map_point(h, p)) == pytest.approx(p, abs=1e-9)


def test_the_inverse_of_the_full_quad_is_the_identity():
    assert inverse_homography(FULL) == pytest.approx(np.eye(3), abs=1e-9)


def test_the_inverse_gives_the_quad_a_positive_w():
    """The shader rejects a fragment by the SIGN of w, so the sign is fixed.

    A projective map has a vanishing line, and display points beyond it map
    back into the unit square with w negated - a ghost of the picture. The
    shader can only tell the ghost from the picture if the quad itself is
    known to be positive.
    """
    for quad in (FULL, SKEW, ((0.02, 0.99), (0.98, 0.60),
                              (0.99, 0.40), (0.01, 0.02))):
        inv = inverse_homography(quad)
        centre = np.mean(np.asarray(quad, dtype=float), axis=0)
        w = inv[2] @ np.array([centre[0], centre[1], 1.0])
        assert w > 0.0, f"{quad} inverted to a negative w"


def test_a_ghost_beyond_the_vanishing_line_has_a_negative_w():
    """What the sign test actually buys, on a quad skewed hard enough."""
    quad = ((0.30, 0.95), (0.70, 0.95), (0.99, 0.05), (0.01, 0.05))
    inv = inverse_homography(quad)

    # Walk away from the quad until w changes sign; the source point there
    # would otherwise land back inside the unit square and draw a ghost.
    ws = [inv[2] @ np.array([0.5, y, 1.0]) for y in np.linspace(0.5, 40.0, 400)]
    assert min(ws) < 0.0, "this quad has no reachable vanishing line"


# --- degenerate quads -----------------------------------------------------

def test_a_bowtie_is_refused():
    """Swapping two corners folds the map; it must not reach the shader."""
    folded = (FULL[0], FULL[1], FULL[3], FULL[2])
    assert not is_convex(folded)
    assert inverse_homography(folded) is None


def test_a_collapsed_edge_is_refused():
    degenerate = (FULL[0], FULL[0], FULL[2], FULL[3])
    assert inverse_homography(degenerate) is None


def test_a_quad_squashed_to_a_line_is_refused():
    flat = ((0.0, 0.5), (1.0, 0.5), (1.0, 0.5), (0.0, 0.5))
    assert inverse_homography(flat) is None


def test_an_ordinary_warp_is_not_refused():
    assert is_convex(SKEW)
    assert inverse_homography(SKEW) is not None


def test_convexity_does_not_depend_on_winding():
    assert is_convex(FULL)
    assert is_convex(tuple(reversed(FULL)))


# --- the handle hit test --------------------------------------------------

def test_the_nearest_corner_is_the_one_the_pointer_is_on():
    for i, c in enumerate(SKEW):
        assert nearest_corner(c, SKEW) == i


def test_a_pointer_between_corners_names_the_closest_one():
    assert nearest_corner((0.80, 0.70), SKEW) == 1      # TR


# --- what goes to disk ----------------------------------------------------

def test_corners_round_trip_through_json_shapes():
    from services.corner_pin import corners_from_json, corners_to_json

    stored = corners_to_json(SKEW)
    assert stored == [[c[0], c[1]] for c in SKEW]
    assert corners_from_json(stored) == pytest.approx(np.array(SKEW))


def test_a_corrupt_stored_calibration_reads_as_none():
    """preferences.config is a file a user can edit. It must not crash."""
    from services.corner_pin import corners_from_json

    for bad in (None, [], "nonsense", [[0, 0]] * 3, [[0, 0]] * 5,
                [[0, 0, 0]] * 4, [["a", "b"]] * 4, {"tl": [0, 0]},
                [[0.0, 0.0]] * 4,                       # collapsed
                [[float("nan"), 0.0], [1, 0], [1, 1], [0, 1]]):
        assert corners_from_json(bad) is None, bad


def test_a_stored_calibration_survives_being_a_list_of_lists():
    from services.corner_pin import corners_from_json

    got = corners_from_json([[0.1, 0.9], [0.9, 0.9], [0.9, 0.1], [0.1, 0.1]])
    assert got == ((0.1, 0.9), (0.9, 0.9), (0.9, 0.1), (0.1, 0.1))
