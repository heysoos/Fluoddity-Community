"""The viewer's diagnostics, on fields whose answer is known by construction.

`best_1d` is the check that decides whether a sweep's two axes were worth
running, so it is worth more than a smoke test: each case below is a field
whose intrinsic dimensionality is not in doubt.
"""
from __future__ import annotations

import numpy as np
import pytest

view = pytest.importorskip("tools.phase_view")

R = 48
XS = np.linspace(-1.0, 1.0, R)
YS = np.linspace(-1.0, 1.0, R)
DONE = np.ones((R, R), dtype=bool)
X, Y = np.meshgrid(XS, YS)


def score(plane):
    return view.best_1d(plane, DONE, XS, YS)


def test_a_field_varying_in_one_axis_only_is_linear():
    r2, label = score(np.sin(3.0 * X))
    assert r2 > 0.98
    assert label.startswith("linear")


def test_a_field_constant_along_rays_is_polar_not_linear():
    """The case the (axial, lateral) sweep turned out to be: the two axes are a
    magnitude and a ratio, so only the ratio was ever swept. A linear ridge
    cannot see this - constant along RAYS is not constant along parallel lines.
    """
    r2, label = score(np.arctan2(Y, X))
    assert label == "polar"
    assert r2 > 0.95


def test_a_genuinely_two_dimensional_field_scores_low():
    """A product of two independent one-dimensional patterns has no single
    coordinate, which is what a plane worth sweeping looks like."""
    r2, _ = score(np.sin(4.0 * X) * np.sin(4.0 * Y))
    assert r2 < 0.6


def test_a_constant_field_is_flat_at_any_magnitude():
    """The guard must be relative: alive_steps sits at 2500 and would clear any
    absolute variance floor set for a channel that lives in 0..1."""
    for level in (0.5, 2500.0):
        r2, label = score(np.full((R, R), level))
        assert label == "flat" and not np.isfinite(r2)


def test_polar_is_declined_when_the_origin_is_outside_the_sweep():
    """Rays only fan out if the sweep contains the point they come from; off to
    one side they are near-parallel and 'polar' would mean nothing."""
    xs = np.linspace(2.0, 3.0, R)
    x, y = np.meshgrid(xs, YS)
    r2, label = view.best_1d(np.arctan2(y, x), DONE, xs, YS)
    assert label.startswith("linear")
    assert np.isfinite(r2)


def bifurcation_sweep(col, y_span=0.0):
    """A sweep whose rows are repeats: `col` is (reps, parameter values)."""
    reps, n = col.shape
    feats = np.zeros((reps, n, 2), dtype=np.float32)
    feats[..., 0] = col
    data = {"cell_names": np.array(["change_rate", "coverage"]),
            "cell_features": feats,
            "done": np.ones((reps, n), bool),
            "y_values": np.linspace(0.0, y_span, reps)}
    meta = {"y_param": "LATERAL_FORCE", "y_range": [0.0, y_span]}
    return data, meta


def test_bifurcation_refuses_a_sweep_whose_rows_are_a_second_parameter():
    """Its whole premise is that a column's rows differ only by the splat
    race. Over a real second axis the spread would be a parameter's doing and
    the picture would read as chaos that is not there."""
    data, meta = bifurcation_sweep(np.zeros((16, 8)), y_span=1.0)
    with pytest.raises(SystemExit, match="collapsed"):
        view.bifurcation(data, meta, "change_rate", 64, 1)


def test_a_reproducible_column_is_thinner_than_a_chaotic_one():
    rng = np.random.default_rng(0)
    col = np.zeros((64, 2))
    col[:, 0] = 0.5 + rng.normal(0, 0.002, 64)   # reproducible
    col[:, 1] = 0.5 + rng.normal(0, 0.150, 64)   # amplified perturbation
    data, meta = bifurcation_sweep(col)
    img, title = view.bifurcation(data, meta, "change_rate", 64, 1)
    assert title == "bifurcation-change_rate"
    assert img.shape == (64, 2, 3)
    lit = (img.max(axis=2) > 120).sum(axis=0)
    assert lit[0] < lit[1], "the chaotic column must occupy more of the axis"


def test_incomplete_cells_are_excluded_rather_than_read_as_zero():
    plane = np.sin(3.0 * X).copy()
    done = DONE.copy()
    done[: R // 2] = False
    plane[: R // 2] = np.nan
    r2, _ = view.best_1d(plane, done, XS, YS)
    assert np.isfinite(r2) and r2 > 0.98
