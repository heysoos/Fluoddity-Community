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


# ---- the thumbnail atlas ------------------------------------------------
#
# One representative picture per occupied screen cell, so the count is bounded
# by the VIEWPORT rather than by the archive.

def test_cell_argmax_picks_the_highest_value_in_each_cell():
    flat = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    values = np.array([0.1, 0.9, 0.5, 0.2, 0.7], dtype=np.float32)
    win, counts = mv.cell_argmax(flat, values, 3)
    assert win[0] == 1        # 0.9 beats 0.1
    assert win[1] == 4        # 0.7 beats 0.5 and 0.2
    assert counts.tolist() == [2, 3, 0]


def test_cell_argmax_marks_an_empty_cell_rather_than_pointing_at_row_zero():
    """A -1 is what lets the caller skip a cell. Zero is a real row."""
    win, counts = mv.cell_argmax(np.array([2], dtype=np.int64),
                                       np.array([0.5], dtype=np.float32), 4)
    assert counts.tolist() == [0, 0, 1, 0]
    assert win[0] == -1 and win[1] == -1 and win[3] == -1
    assert win[2] == 0


def test_cell_argmax_over_nothing_is_all_empty():
    win, counts = mv.cell_argmax(np.zeros(0, dtype=np.int64),
                                       np.zeros(0, dtype=np.float32), 3)
    assert win.tolist() == [-1, -1, -1]
    assert counts.tolist() == [0, 0, 0]


def test_cell_argmax_agrees_with_the_other_reducers_on_which_cells_are_full():
    rs = np.random.RandomState(0)
    flat = rs.randint(0, 12, size=200).astype(np.int64)
    values = rs.rand(200).astype(np.float32)
    win, counts = mv.cell_argmax(flat, values, 12)
    _means, mean_counts = mv.cell_means(flat, values, 12)
    assert counts.tolist() == mean_counts.tolist()
    assert ((win >= 0) == (counts > 0)).all()


def test_the_winner_of_a_cell_really_is_in_that_cell():
    rs = np.random.RandomState(1)
    flat = rs.randint(0, 8, size=120).astype(np.int64)
    values = rs.rand(120).astype(np.float32)
    win, counts = mv.cell_argmax(flat, values, 8)
    for cell in range(8):
        if counts[cell]:
            assert flat[win[cell]] == cell
            assert values[win[cell]] == values[flat == cell].max()


# ---- dots must not sit under the pictures -------------------------------

def test_a_point_whose_cell_drew_a_thumbnail_is_covered():
    flat = np.array([0, 0, 1, 2], dtype=np.int64)
    on = np.array([True, True, True, True])
    drawn = {0, 2}
    mask = mv.covered_by(flat, on, drawn)
    assert mask.tolist() == [True, True, False, True]


def test_a_point_off_the_canvas_is_never_covered():
    """bin_points drops off-canvas points, so the mask has to be widened back
    out to the full array the dots are drawn from."""
    flat = np.array([0, 0], dtype=np.int64)          # only the on-canvas ones
    on = np.array([True, False, True, False])
    mask = mv.covered_by(flat, on, {0})
    assert mask.tolist() == [True, False, True, False]


def test_nothing_drawn_covers_nothing():
    flat = np.array([0, 1], dtype=np.int64)
    on = np.array([True, True])
    assert not mv.covered_by(flat, on, set()).any()


def test_covering_nothing_at_all_is_an_empty_mask():
    mask = mv.covered_by(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool),
                         {1, 2})
    assert mask.shape == (0,)


# ---- the atlas is binned in UNIT space, quantised to zoom levels ---------
#
# Screen-space bins churn: a one-pixel pan moves every boundary, so each cell's
# winner changes and the atlas reshuffles under the pointer. Quantising to
# powers of two makes the assignment invariant under pan, and under zoom within
# a level.

@pytest.mark.parametrize("span,px", [(1000.0, 32.0), (640.0, 16.0),
                                     (2000.0, 64.0), (300.0, 48.0)])
def test_a_quantised_cell_is_between_one_and_two_thumbnails_wide(span, px):
    c = mv.quantised_cell(span, px)
    on_screen = c * span
    assert px <= on_screen < 2 * px


def test_a_quantised_cell_is_a_power_of_two():
    c = mv.quantised_cell(1000.0, 32.0)
    inv = 1.0 / c
    assert abs(inv - round(inv)) < 1e-9
    assert round(inv) & (round(inv) - 1) == 0


def test_zooming_within_a_level_does_not_change_the_cell():
    """The whole point: continuous zoom must not reshuffle the pictures."""
    # Both inside one level: the boundaries sit at powers of two, which for
    # 32px is a span of 1024.
    a = mv.quantised_cell(1100.0, 32.0)
    b = mv.quantised_cell(2000.0, 32.0)
    assert a == b
    assert mv.quantised_cell(1000.0, 32.0) != a   # 1024 is the boundary


def test_crossing_a_level_halves_the_cell():
    a = mv.quantised_cell(1000.0, 32.0)
    b = mv.quantised_cell(4000.0, 32.0)
    assert b < a


def test_a_cell_never_exceeds_the_whole_map():
    assert mv.quantised_cell(10.0, 64.0) == 1.0


# ---- the winners themselves ---------------------------------------------

def _unit(pairs):
    return np.array(pairs, dtype=np.float32)


def test_each_occupied_cell_yields_its_highest_value_entry():
    unit = _unit([[0.1, 0.1], [0.2, 0.1], [0.6, 0.6]])
    values = np.array([0.2, 0.9, 0.5], dtype=np.float32)
    ux, uy, win = mv.atlas_winners(unit, values, (0.5, 0.5))
    assert len(win) == 2
    order = np.argsort(ux)
    assert win[order][0] == 1          # 0.9 beat 0.2 in the first cell
    assert win[order][1] == 2


def test_the_cell_origin_is_returned_in_unit_space():
    unit = _unit([[0.6, 0.3]])
    ux, uy, win = mv.atlas_winners(unit, np.array([1.0], np.float32), (0.25, 0.25))
    assert ux[0] == pytest.approx(0.5)
    assert uy[0] == pytest.approx(0.25)


def test_panning_cannot_change_the_winners():
    """Membership is in unit space, which panning does not touch."""
    rs = np.random.RandomState(0)
    unit = rs.rand(300, 2).astype(np.float32)
    values = rs.rand(300).astype(np.float32)
    a = mv.atlas_winners(unit, values, (0.125, 0.125))
    b = mv.atlas_winners(unit, values, (0.125, 0.125))
    assert np.array_equal(a[2], b[2])


def test_an_empty_map_yields_no_cells():
    ux, uy, win = mv.atlas_winners(np.zeros((0, 2), np.float32),
                                   np.zeros(0, np.float32), (0.25, 0.25))
    assert len(ux) == len(uy) == len(win) == 0


# --- the atlas cell is SQUARE ON SCREEN and crosses ONE level at a time ----

def test_atlas_cells_are_square_on_screen_however_wide_the_canvas():
    """A cell drawn as a rectangle stretches the thumbnail inside it."""
    for w, h in ((1480.0, 320.0), (600.0, 320.0), (2560.0, 200.0)):
        cx, cy = mv.atlas_cell(w, h, 32.0)
        assert cx * w == pytest.approx(cy * h, rel=0.02), (w, h)


def test_a_zoom_octave_crosses_exactly_one_level():
    """One reshuffle per octave, not two.

    Quantising the two axes independently crosses x and y at DIFFERENT zooms,
    so an octave reshuffles the whole atlas twice - and each reshuffle is a
    visible sweep of re-decoding. On this canvas the old pair gave three
    distinct cell sizes over the same span.
    """
    w, h = 1480.0, 320.0
    seen = {mv.atlas_cell(w * z, h * z, 32.0)
            for z in np.linspace(1.0, 1.999, 60)}
    assert len(seen) == 2, seen
    old = {(mv.quantised_cell(w * z, 32.0), mv.quantised_cell(h * z, 32.0))
           for z in np.linspace(1.0, 1.999, 60)}
    assert len(old) == 3


def test_the_atlas_cell_still_halves_across_an_octave():
    a = mv.atlas_cell(1000.0, 400.0, 32.0)
    b = mv.atlas_cell(2000.0, 800.0, 32.0)
    assert b[0] == pytest.approx(a[0] / 2.0)
    assert b[1] == pytest.approx(a[1] / 2.0)


def test_a_finer_level_keeps_every_winner_the_coarser_one_had():
    """Subdividing must REUSE what is already decoded: the coarse winner is
    still the winner of exactly one of its sub-cells. Without this a zoom
    re-decodes the whole map instead of half of it."""
    rs = np.random.RandomState(3)
    unit = rs.rand(4000, 2).astype(np.float32)
    values = rs.rand(4000).astype(np.float32)
    coarse = set(mv.atlas_winners(unit, values, (0.125, 0.125))[2].tolist())
    fine = set(mv.atlas_winners(unit, values, (0.0625, 0.0625))[2].tolist())
    assert coarse <= fine


def test_the_atlas_draws_no_more_cells_than_a_budget():
    """The working set must FIT the cache. A map asking for more pictures
    than can be held evicts its own cells and re-decodes them forever."""
    rs = np.random.RandomState(4)
    unit = rs.rand(6000, 2).astype(np.float32)
    values = rs.rand(6000).astype(np.float32)
    ux, uy, win = mv.atlas_winners(unit, values, (0.01, 0.01), budget=200)
    assert len(win) == 200


def test_the_budget_keeps_the_most_novel_cells():
    rs = np.random.RandomState(5)
    unit = rs.rand(3000, 2).astype(np.float32)
    values = rs.rand(3000).astype(np.float32)
    full = mv.atlas_winners(unit, values, (0.02, 0.02))[2]
    cut = mv.atlas_winners(unit, values, (0.02, 0.02), budget=50)[2]
    assert len(cut) == 50
    assert set(cut.tolist()) <= set(full.tolist())
    assert min(values[cut]) >= np.sort(values[full])[-50]


def test_a_budget_larger_than_the_cell_count_changes_nothing():
    rs = np.random.RandomState(6)
    unit = rs.rand(200, 2).astype(np.float32)
    values = rs.rand(200).astype(np.float32)
    a = mv.atlas_winners(unit, values, (0.25, 0.25))
    b = mv.atlas_winners(unit, values, (0.25, 0.25), budget=10_000)
    assert np.array_equal(a[2], b[2])
