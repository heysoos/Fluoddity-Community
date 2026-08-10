"""Filtering, colouring and density binning for the archive map.

Pure over plain entry data, so the rules are assertable without ImGui.
"""
import numpy as np
import pytest

from services import map_view as mv


class _E:
    def __init__(self, i, gen=0, novelty=0.5, liveness=0.02, source="expansion",
                 goal="", pinned=False):
        self.id = i
        self.gen = gen
        self.novelty = novelty
        self.liveness = liveness
        self.source = source
        self.goal = goal
        self.pinned = pinned


def _entries(n=100):
    return [_E(i, gen=i, novelty=i / n, liveness=0.001 * i,
               source=("expedition" if i % 3 else "expansion"),
               goal=("flowers" if i % 5 == 0 else ""))
            for i in range(n)]


# ---- defaults keep the old map -------------------------------------------

def test_the_defaults_are_the_historical_view():
    """These are additions; the plain scatter must stay one combo away."""
    assert mv.COLOR_MODES[0] == "source"
    assert mv.FILTER_MODES[0] == "all"
    assert mv.RENDER_MODES[0] == "points"


def test_the_all_filter_keeps_everything_in_order():
    e = _entries()
    got = mv.filter_indices(e, "all")
    assert got.tolist() == list(range(len(e)))


def test_colouring_by_source_asks_for_no_ramp():
    assert mv.scalar_values(_entries(), "source") is None


# ---- filters --------------------------------------------------------------

def test_recent_keeps_the_last_n_generations():
    e = _entries(100)                       # gen == index
    got = mv.filter_indices(e, "recent", recent_gens=10)
    assert got.tolist() == list(range(89, 100))


def test_novel_keeps_the_top_percent_by_novelty():
    e = _entries(100)                       # novelty ascending with index
    got = mv.filter_indices(e, "novel", novel_pct=25)
    assert len(got) == 25
    assert got.tolist() == list(range(75, 100))


def test_kept_is_the_results_of_the_search():
    e = _entries(10)
    e[2].source = "summit"
    e[5].source = "record"
    e[7].pinned = True
    assert mv.filter_indices(e, "kept").tolist() == [2, 5, 7]


def test_goal_and_source_filters_select_exactly():
    e = _entries(20)
    assert all(e[i].goal == "flowers"
               for i in mv.filter_indices(e, "goal", goal="flowers"))
    assert all(e[i].source == "expansion"
               for i in mv.filter_indices(e, "source", source="expansion"))


def test_indices_are_always_ascending():
    """The hover lookup maps row -> archive index; a shuffled mapping would
    select the wrong entry."""
    e = _entries(50)
    for mode in mv.FILTER_MODES:
        got = mv.filter_indices(e, mode, goal="flowers", source="expedition")
        assert np.all(np.diff(got) > 0) or len(got) < 2


def test_an_unknown_filter_shows_everything_rather_than_nothing():
    """A stale persisted setting must not present as an empty map."""
    e = _entries(10)
    assert len(mv.filter_indices(e, "nonsense-from-an-old-settings-file")) == 10


def test_filters_survive_an_empty_archive():
    for mode in mv.FILTER_MODES:
        assert len(mv.filter_indices([], mode)) == 0


def test_a_filter_matching_nothing_returns_empty():
    e = _entries(10)
    assert len(mv.filter_indices(e, "goal", goal="no such goal")) == 0


@pytest.mark.parametrize("pct", [1, 25, 100])
def test_novel_always_keeps_at_least_one(pct):
    assert len(mv.filter_indices(_entries(7), "novel", novel_pct=pct)) >= 1


# ---- the colour ramp ------------------------------------------------------

def test_normalise_spans_zero_to_one():
    t = mv.normalise(np.linspace(0, 1, 100))
    assert t.min() == pytest.approx(0.0) and t.max() == pytest.approx(1.0)


def test_normalise_resists_a_single_outlier():
    """Generation 0 entries are stamped novelty 1.0 by the no-reference
    convention; on min/max bounds that one entry flattens everything else."""
    bulk = np.linspace(0.01, 0.05, 99).astype(np.float32)
    t = mv.normalise(np.concatenate([bulk, [1.0]]))

    assert t[-1] == pytest.approx(1.0), "the outlier still reads as the top"
    # The point: the real entries keep the ramp. On min/max bounds they would
    # span (0.05 - 0.01) / (1.0 - 0.01) = 4% of it and all look identical.
    assert t[:99].max() - t[:99].min() > 0.9


def test_normalise_flattens_a_degenerate_distribution_rather_than_dividing():
    """99 identical values and one outlier leaves no robust range at all; the
    honest answer is a flat mid-ramp, not a divide by ~0."""
    v = np.concatenate([np.full(99, 0.02, np.float32), [1.0]])
    t = mv.normalise(v)
    assert np.all(t == 0.5) and np.isfinite(t).all()


def test_normalise_handles_a_constant_column():
    t = mv.normalise(np.full(10, 0.3, np.float32))
    assert np.all(t == 0.5) and np.isfinite(t).all()


def test_the_ramp_is_monotone_in_lightness():
    """It has to read as an ordering, not just as different colours."""
    c = mv.ramp_colors(np.linspace(0, 1, 32))
    lum = [(x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) for x in c.tolist()]
    assert all(b >= a - 12 for a, b in zip(lum, lum[1:])), lum


def test_the_ramp_is_visible_against_the_canvas():
    """The canvas is (20, 20, 24); a low end darker than that is invisible."""
    lo = int(mv.ramp_colors(np.array([0.0]))[0])
    assert (lo & 255) + ((lo >> 8) & 255) + ((lo >> 16) & 255) > 120


def test_the_ramp_packs_the_requested_alpha():
    assert (int(mv.ramp_colors(np.array([0.5]), alpha=200)[0]) >> 24) & 255 == 200


def test_the_ramp_survives_an_empty_selection():
    assert len(mv.ramp_colors(np.zeros(0))) == 0


# ---- density --------------------------------------------------------------

def test_density_counts_land_in_the_right_cells():
    xs = np.array([0.0, 1.0, 50.0], dtype=np.float32)
    ys = np.array([0.0, 1.0, 50.0], dtype=np.float32)
    counts, nx, ny, cell = mv.density_grid(xs, ys, (0, 0), (100, 100), cell_px=10)
    assert counts.sum() == 3
    assert counts[0, 0] == 2          # the two inside the first 10x10 cell
    assert counts[5, 5] == 1


def test_density_ignores_points_outside_the_canvas():
    xs = np.array([-5.0, 105.0, 50.0], dtype=np.float32)
    ys = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    counts, *_ = mv.density_grid(xs, ys, (0, 0), (100, 100), cell_px=10)
    assert counts.sum() == 1


def test_density_is_binned_in_screen_space():
    """Zooming must raise the resolution; a fixed grid in projection space
    would turn back into one blob the moment you zoom in."""
    xs = np.array([10.0, 12.0], dtype=np.float32)
    ys = np.array([10.0, 12.0], dtype=np.float32)
    coarse, *_ = mv.density_grid(xs, ys, (0, 0), (100, 100), cell_px=10)
    fine, *_ = mv.density_grid(xs, ys, (0, 0), (100, 100), cell_px=2)
    assert coarse.max() == 2 and fine.max() == 1


def test_density_intensity_is_logarithmic():
    """Archive density spans orders of magnitude - linear renders the whole
    frontier as empty, and the frontier is the half worth seeing."""
    counts = np.array([[1, 10, 100, 1000]])
    t = mv.density_intensity(counts)
    assert all(s > 0 for s in np.diff(t[0])), "still monotone in count"

    # The property that matters: against a linear ramp, a cell holding one
    # entry is 0.001 and invisible. Here it is ~0.10.
    linear = counts[0] / counts.max()
    assert t[0, 0] > 0.09
    assert t[0, 0] > 100 * linear[0]
    assert t[0, 1] > 10 * linear[1]


def test_density_survives_an_empty_canvas():
    counts, nx, ny, cell = mv.density_grid(
        np.zeros(0), np.zeros(0), (0, 0), (100, 100))
    assert counts.sum() == 0 and counts.shape == (ny, nx)
    assert np.all(mv.density_intensity(counts) == 0)


def test_a_degenerate_canvas_does_not_divide_by_zero():
    counts, nx, ny, _ = mv.density_grid(
        np.array([1.0]), np.array([1.0]), (0, 0), (0, 0))
    assert nx >= 1 and ny >= 1 and np.isfinite(counts).all()


# ---- combo population ----------------------------------------------------

def test_present_values_lists_what_the_archive_actually_holds():
    e = _entries(20)
    assert mv.present_values(e, "goal") == ["flowers"]
    assert mv.present_values(e, "source") == ["expansion", "expedition"]


def test_present_values_drops_the_empty_string():
    assert "" not in mv.present_values(_entries(20), "goal")


# ---- density carries the colour mode -------------------------------------

def _grid_of(xs, ys, cell=10.0):
    return mv.bin_points(np.asarray(xs, np.float32), np.asarray(ys, np.float32),
                         (0, 0), (100, 100), cell_px=cell)


def test_bin_points_and_density_grid_agree():
    xs = np.array([1.0, 3.0, 55.0], np.float32)
    ys = np.array([1.0, 2.0, 55.0], np.float32)
    flat, on, nx, ny, cell = _grid_of(xs, ys)
    counts, gnx, gny, gcell = mv.density_grid(xs, ys, (0, 0), (100, 100), 10.0)
    assert (nx, ny, cell) == (gnx, gny, gcell)
    assert counts.reshape(-1)[flat[0]] == 2
    assert on.tolist() == [True, True, True]


def test_bin_points_drops_what_is_off_canvas():
    flat, on, *_ = _grid_of([-5.0, 50.0], [50.0, 50.0])
    assert on.tolist() == [False, True]
    assert len(flat) == 1


def test_cell_means_average_the_entries_that_landed_there():
    """Colouring a heatmap by novelty means the cell shows the novelty of what
    is in it, not how many things are in it."""
    flat = np.array([0, 0, 3], dtype=np.int64)
    vals = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    means, counts = mv.cell_means(flat, vals, 4)
    assert means[0] == pytest.approx(0.5)
    assert means[3] == pytest.approx(0.5)
    assert counts.tolist() == [2, 0, 0, 1]


def test_an_empty_cell_has_no_mean_rather_than_a_nan():
    means, counts = mv.cell_means(np.zeros(0, np.int64),
                                  np.zeros(0, np.float32), 4)
    assert np.isfinite(means).all() and counts.sum() == 0


def test_cell_majority_picks_a_winner_rather_than_blending():
    """Averaging two packed colours is not a colour - it is whatever bit
    pattern falls out of the arithmetic."""
    flat = np.array([0, 0, 0, 1], dtype=np.int64)
    codes = np.array([111, 111, 222, 222], dtype=np.int64)
    best, counts = mv.cell_majority(flat, codes, 2)
    assert best[0] == 111 and best[1] == 222
    assert counts.tolist() == [3, 1]


def test_cell_majority_leaves_empty_cells_alone():
    best, counts = mv.cell_majority(np.array([2], np.int64),
                                    np.array([777], np.int64), 4)
    assert best.tolist() == [0, 0, 777, 0]
    assert counts.tolist() == [0, 0, 1, 0]


def test_density_alpha_rises_with_count_and_never_vanishes():
    """Count moves to the alpha channel once colour is carrying something
    else; a one-entry cell that fades to nothing hides the frontier."""
    a = mv.density_alpha(np.array([1, 5, 50, 500]))
    assert list(a) == sorted(a)
    assert a[0] >= mv.DENSITY_ALPHA_MIN
    assert a[-1] <= 255


def test_density_alpha_survives_an_empty_grid():
    assert np.isfinite(mv.density_alpha(np.zeros(4, np.int32))).all()


def test_ramp_takes_a_per_entry_alpha():
    cols = mv.ramp_colors(np.array([0.5, 0.5], np.float32),
                          alpha=np.array([10, 250]))
    assert [(int(c) >> 24) & 255 for c in cols] == [10, 250]
    assert (int(cols[0]) & 0xFFFFFF) == (int(cols[1]) & 0xFFFFFF)


def test_with_alpha_replaces_only_the_alpha_byte():
    src = mv.pack(10, 20, 30, 255)
    got = int(mv.with_alpha(np.array([src]), 77)[0])
    assert got & 0xFFFFFF == src & 0xFFFFFF
    assert (got >> 24) & 255 == 77
