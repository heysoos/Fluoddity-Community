"""Actually render the Explore tab and the archive browser.

An ImGui begin/end imbalance or a None dereference does not fail loudly - it
corrupts the whole frame, so every window in the app disappears at once and the
cause is invisible. ImGui asserts on an unbalanced stack inside EndFrame, so a
test that completes at all has proved the stack balances.

No window and no renderer: ImGui only needs a display size and a frame.
"""
import numpy as np
import pytest
from imgui_bundle import imgui

from services.goal_source import GoalList
from state.archive_state import ArchiveState
from state.sim_state import SimState
from state.auto_tournament_state import AutoTournamentState
from state.tournament_state import TournamentState
from ui import layout
from ui.archive_window import ArchiveWindowMixin
from ui.auto_tournament_window import AutoTournamentWindowMixin


@pytest.fixture(scope="module")
def gui():
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 900)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    for _ in range(2):
        imgui.new_frame()
        imgui.render()
    yield
    imgui.destroy_context()


def frame(fn, n=2):
    """Run full ImGui frames around fn and return the last vertex count."""
    for _ in range(n):
        imgui.new_frame()
        imgui.begin("host", True)
        fn()
        imgui.end()
        imgui.render()
    return imgui.get_draw_data().total_vtx_count


def host_only():
    """Vertices for the bare host window.

    'Rendered nothing' is not zero: an empty ImGui window still has a title bar
    and a background, measured at 36 vertices. Comparing against this baseline
    is what makes "the hidden window drew nothing" an actual assertion.
    """
    return frame(lambda: None)


class _State:
    def __init__(self):
        self.archive = ArchiveState()
        # The Auto tab reads it for the sweep warning.
        self.sim = SimState()
        self.auto_tournament = AutoTournamentState()
        self.tournament = TournamentState()


class _FakeArchive:
    def __init__(self, n=0, dim=8):
        self.entries = []
        self.embeddings = np.zeros((n, dim), dtype=np.float32)
        # The real Archive carries this and the browser caches against it. A
        # fake without one would silently exercise the getattr fallback instead
        # of the path that actually ships.
        self.revision = 0
        self.signature = "fourier-n10"

    def layout_at(self, i):
        return getattr(self.entries[i], "layout", "") or self.signature

    def is_native(self, i):
        return self.layout_at(i) == self.signature

    def thumb_key(self, i):
        t = getattr(self.entries[i], "thumb", "")
        return f"{self.layout_at(i)}/{t}" if t else ""

    def __len__(self):
        return len(self.entries)

    def stats(self):
        return {"size": len(self.entries), "capacity": 20000,
                "admission_rate": 0.98, "n_nonfinite": 0, "n_rejected": 0,
                "n_evicted": 0, "n_pinned": 0, "blocked_by_pins": False,
                "rejects_ring": 0}


class _FakeDriver:
    def __init__(self, **over):
        self._st = {"regime": "expansion", "goal": "coral reef",
                    "archive_size": 3, "capacity": 20000, "n_evicted": 0,
                    "admission_rate": 0.98,
                    "n_pinned": 0, "blocked_by_pins": False,
                    "score_label": "novelty", "sigma": 0.15,
                    "algorithm": "CMA-ES", "prompt": "coral reef"}
        self._st.update(over)

    def status(self):
        return dict(self._st)


class _TracingDriver(_FakeDriver):
    """A driver that reports the phase and the traces, so the progress bar and
    both plots are actually drawn rather than skipped by their .get() guards."""

    def __init__(self, gens=40, **over):
        super().__init__(**over)
        self._st.update(
            phase={"label": "expedition: coral reef", "done": 12, "total": 50,
                   "unit": "generations", "note": "38 left"},
            last_tiles=64, last_admitted=3,
            n_rejected_close=900, n_rejected_dead=7,
            min_separation=0.02, mean_novelty=0.031)
        self.trace = {
            "gen": list(range(gens)),
            # Half the run outside an expedition, so the NaN break in the
            # fitness line is exercised too.
            "regime": ["expansion"] * (gens // 2) + ["expedition"] * (gens - gens // 2),
            "archive_size": [100 + 3 * i for i in range(gens)],
            "mean_novelty": [0.05 - 0.0005 * i for i in range(gens)],
            "admitted": [3] * gens,
            "tiles": [64] * gens,
            "fit_best": [float("nan")] * (gens // 2)
                        + [0.2 + 0.01 * i for i in range(gens - gens // 2)],
            "fit_mean": [float("nan")] * (gens // 2)
                        + [0.1 + 0.005 * i for i in range(gens - gens // 2)],
        }

    def expedition_trace(self):
        n = self.trace["regime"].count("expedition")
        return {"best": self.trace["fit_best"][-n:] if n else [],
                "mean": self.trace["fit_mean"][-n:] if n else [],
                "gens": n}


class Harness(ArchiveWindowMixin, AutoTournamentWindowMixin):
    """The two mixins under test, with only the attributes they reach for."""

    def __init__(self, driver=None, archive=None, goals=None, unavailable=""):
        self.state = _State()
        self.archive_driver = driver
        self.archive_obj = archive
        self.archive_goals = goals
        self.archive_service = None
        self.archive_unavailable = unavailable
        self.archive_projection = None
        self.archive_goal_point = None
        self.thumb_cache = None


def test_explore_tab_renders_before_the_service_exists(gui):
    h = Harness()
    assert frame(h.render_explore_tab) > host_only()


def test_explore_tab_renders_when_clip_is_unavailable(gui):
    h = Harness(unavailable="missing package: onnxruntime")
    assert frame(h.render_explore_tab) > host_only()


def test_explore_tab_renders_with_a_live_driver(gui):
    goals = GoalList()
    goals.add("coral reef")
    goals.add("lightning")
    h = Harness(driver=_FakeDriver(), archive=_FakeArchive(), goals=goals)
    assert frame(h.render_explore_tab) > host_only()


def test_the_phase_bar_and_both_traces_render(gui):
    """The status block used to be four lines of text that read identically
    whether the search was running, stalled or stopped."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive())
    h.state.archive.running = True
    assert frame(h.render_explore_tab) > host_only()


def test_the_traces_render_before_any_generation_has_run(gui):
    d = _TracingDriver(gens=0)
    d.trace = {k: [] for k in d.trace}
    h = Harness(driver=d, archive=_FakeArchive())
    assert frame(h.render_explore_tab) > host_only()


def test_a_phase_with_no_finish_line_renders(gui):
    """Expansion Between = 0. A progress bar needs an end; there is none."""
    d = _TracingDriver()
    d._st["phase"] = {"label": "expansion: no expeditions", "done": 0,
                      "total": 0, "unit": "", "note": ""}
    h = Harness(driver=d, archive=_FakeArchive())
    assert frame(h.render_explore_tab) > host_only()


def test_explore_tab_warns_when_the_archive_is_full_of_pins(gui):
    h = Harness(driver=_FakeDriver(blocked_by_pins=True, archive_size=10),
                archive=_FakeArchive())
    assert frame(h.render_explore_tab) > host_only()


def test_explore_tab_warns_when_nothing_is_being_admitted(gui):
    h = Harness(driver=_FakeDriver(archive_size=500, admission_rate=0.0),
                archive=_FakeArchive())
    assert frame(h.render_explore_tab) > host_only()


def test_rendering_the_tab_marks_the_mode_enabled(gui):
    h = Harness()
    frame(h.render_explore_tab)
    assert h.state.archive.enabled is True


def test_a_dismissable_warning_renders(gui):
    h = Harness()
    h.state.archive.warning = "capture is nearly black"
    assert frame(h.render_explore_tab) > host_only()


def test_the_archive_window_is_hidden_until_asked_for(gui):
    h = Harness(archive=_FakeArchive())
    assert frame(h.render_archive_window) == host_only()


def test_the_archive_window_renders_with_no_archive(gui):
    h = Harness()
    h.state.archive.show_browser = True
    assert frame(h.render_archive_window) > host_only()


# ---- the gallery -------------------------------------------------------

def _populated(n=8):
    """A real ArchiveEntry per slot, so the gallery walks real fields."""
    from services.archive import ArchiveEntry

    arc = _FakeArchive(n=n)
    for i in range(n):
        arc.entries.append(ArchiveEntry(
            id=i, novelty=0.5 - 0.01 * i, liveness=0.02 * (i + 1),
            pinned=(i % 4 == 0),
            source=("expedition" if i % 3 else "expansion"),
            spec="brain:80", goal=("coral reef" if i % 2 else ""),
            run_id="r", gen=i, tile=i % 4, ts=1000.0 + i,
            thumb=f"{i:06d}.jpg"))
    return arc


class _FakeTex:
    """imgui.ImTextureRef only needs the integer handle; nothing dereferences
    it until the renderer runs, and these tests never get that far."""
    glo = 1


def test_the_gallery_renders_entries(gui):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    assert frame(h.render_archive_window) > host_only()


def test_the_gallery_renders_thumbnails_when_the_cache_has_them(gui):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True

    class _Cache:
        def get(self, name):
            return _FakeTex()

    h.thumb_cache = _Cache()
    assert frame(h.render_archive_window) > host_only()


def test_a_missing_thumbnail_falls_back_to_a_placeholder(gui):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True

    class _Cache:
        def get(self, name):
            return None

    h.thumb_cache = _Cache()
    assert frame(h.render_archive_window) > host_only()


@pytest.mark.parametrize("mode", ["novelty", "recency", "liveness", "nonsense"])
def test_every_sort_mode_renders(gui, mode):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    h.state.archive.sort_by = mode
    assert frame(h.render_archive_window) > host_only()


def test_the_pinned_only_filter_renders(gui):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    h.state.archive.pinned_only = True
    assert frame(h.render_archive_window) > host_only()


def button_labels(fn, n=3):
    """Every button label `fn` emits.

    Vertex counts cannot answer this: ImGui culls geometry for content below
    the fold, and the shared module context means this window's size is
    whatever an earlier test left it at. Our Python code runs either way, so
    the labels are the honest signal.
    """
    seen = []
    real = imgui.button

    def spy(label, *a, **kw):
        seen.append(label)
        return real(label, *a, **kw)

    imgui.button = spy
    try:
        frame(fn, n=n)
    finally:
        imgui.button = real
    return seen


def _button_labels(h, n=3):
    return button_labels(h.render_archive_window, n=n)


def invisible_button_labels(fn, n=3):
    """Every invisible_button id `fn` emits. The map's canvas and its recentre
    icon are both invisible buttons - the icon is DRAWN, so there is no label
    for button_labels to catch."""
    seen = []
    real = imgui.invisible_button

    def spy(str_id, *a, **kw):
        seen.append(str_id)
        return real(str_id, *a, **kw)

    imgui.invisible_button = spy
    try:
        frame(fn, n=n)
    finally:
        imgui.invisible_button = real
    return seen


def checkbox_labels(fn, n=3):
    """Every checkbox label `fn` emits. Same reasoning as button_labels: a
    disabled or below-the-fold widget still runs our Python."""
    seen = []
    real = imgui.checkbox

    def spy(label, *a, **kw):
        seen.append(label)
        return real(label, *a, **kw)

    imgui.checkbox = spy
    try:
        frame(fn, n=n)
    finally:
        imgui.checkbox = real
    return seen


ACTIONS = {"Save as config...", "Seed a run from here", "Delete"}


def test_the_actions_are_hidden_until_an_entry_is_selected(gui):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    assert not (ACTIONS & set(_button_labels(h)))


def test_selecting_an_entry_reveals_the_actions(gui):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    h.state.archive.selected_entry_id = 3
    assert ACTIONS <= set(_button_labels(h))


def test_every_entry_gets_a_placeholder_when_there_are_no_thumbnails(gui):
    h = Harness(archive=_populated(n=8))
    h.state.archive.show_browser = True
    # a set: _button_labels runs several frames, so every label repeats.
    # Split on "##": everything after it is ImGui id, not visible text - the
    # gallery needs one because two brains each hold an entry #0.
    labels = {x.split("##")[0] for x in _button_labels(h) if x.startswith("#")}
    assert labels == {f"#{i}" for i in range(8)}


def test_sorting_orders_entries_as_labelled(gui):
    """The combo labels promise an order; this is the only thing that checks
    the labels and the sort keys agree."""
    h = Harness(archive=_populated())
    ast = h.state.archive

    ast.sort_by = "novelty"
    nov = [e.novelty for _, e in h._sorted_entries(ast, h.archive_obj)]
    assert nov == sorted(nov, reverse=True)

    ast.sort_by = "liveness"
    liv = [e.liveness for _, e in h._sorted_entries(ast, h.archive_obj)]
    assert liv == sorted(liv, reverse=True)

    ast.sort_by = "recency"
    ts = [e.ts for _, e in h._sorted_entries(ast, h.archive_obj)]
    assert ts == sorted(ts, reverse=True)

    ast.sort_by = "nonsense"
    fallback = [e.novelty for _, e in h._sorted_entries(ast, h.archive_obj)]
    assert fallback == sorted(fallback, reverse=True), "unknown mode falls back"


def test_the_pinned_only_filter_keeps_only_pins(gui):
    h = Harness(archive=_populated())
    h.state.archive.pinned_only = True
    got = h._sorted_entries(h.state.archive, h.archive_obj)
    assert got and all(e.pinned for _, e in got)


# ---- the map -----------------------------------------------------------

def _spread(arc, seed=0):
    """Give the fake archive embeddings with real structure, and a projection
    fitted to them."""
    from services.archive_projection import Projection

    rng = np.random.default_rng(seed)
    e = rng.normal(size=(len(arc.entries), 8)).astype(np.float32)
    e[:, 0] *= 6.0
    e[:, 1] *= 3.0
    arc.embeddings = e / np.linalg.norm(e, axis=1, keepdims=True)
    proj = Projection()
    proj.fit(arc.embeddings)
    return proj


def test_the_map_says_so_when_there_is_nothing_to_project(gui):
    """The guard message renders, but the scatter does not - "Refit projection"
    only exists on the drawing path, so it is the precise signal."""
    h = Harness(archive=_populated(n=2))
    h.archive_projection = _spread(h.archive_obj)
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Refit projection" not in labels


def test_an_unfitted_projection_does_not_stack_every_point_at_the_origin(gui):
    """Projection.transform returns zeros before it is fitted. Rendering that
    would pile the whole archive in one corner and look like a bug rather than
    an unbuilt map."""
    from services.archive_projection import Projection

    h = Harness(archive=_populated())
    h.archive_projection = Projection()          # never fitted
    assert h.archive_projection.fitted is False
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Refit projection" not in labels


def test_a_fitted_projection_does_draw_the_scatter(gui):
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Refit projection" in labels


def test_the_map_renders_a_fitted_archive(gui):
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    assert frame(lambda: h._render_map(h.state.archive, h.archive_obj)) > host_only()


def test_the_map_draws_the_goal_marker(gui):
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    without = frame(lambda: h._render_map(h.state.archive, h.archive_obj))
    h.archive_goal_point = h.archive_obj.embeddings[0].copy()
    assert frame(lambda: h._render_map(h.state.archive, h.archive_obj)) > without


def test_every_entry_source_has_a_colour(gui):
    """A source with no colour silently falls back to the expansion green, so
    expedition points would be indistinguishable from expansion ones."""
    assert set(Harness._MAP_COLORS) >= {"bootstrap", "expansion", "expedition", "pin"}


def test_the_map_renders_through_the_archive_window(gui):
    """The Map tab is only reached via the tab bar, which is where a begin/end
    imbalance in _render_map would corrupt the frame."""
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.show_browser = True
    assert frame(h.render_archive_window) > host_only()


def test_the_map_prompts_you_to_click_when_nothing_is_selected(gui):
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Save as config...##map" not in labels


def test_clicking_a_dot_shows_that_entrys_image_and_actions(gui):
    """A dot is a position; without the picture the map says where something
    sits but never what it is."""
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.selected_entry_id = 5
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Save as config...##map" in labels
    assert "Delete##map" in labels


def test_the_map_selection_falls_back_to_a_placeholder_without_a_thumbnail(gui):
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.selected_entry_id = 5

    class _Cache:
        def get(self, name):
            return None

    h.thumb_cache = _Cache()
    assert "#5##mapsel" in button_labels(
        lambda: h._render_map(h.state.archive, h.archive_obj))


def test_a_stale_selection_does_not_break_the_map(gui):
    """The entry may have been deleted since it was picked."""
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.selected_entry_id = 9999
    assert frame(lambda: h._render_map(h.state.archive, h.archive_obj)) > host_only()


# ---- the archive picker row --------------------------------------------

from ui.archive_window import archive_row_model, new_archive_status  # noqa: E402

TWO = [{"name": "default", "entries": 1511, "mtime": 0.0, "size_mb": 214.0},
       {"name": "dense-trails", "entries": 12, "mtime": 0.0, "size_mb": 1.5}]


def _ast(names=(), active="default", **over):
    ast = ArchiveState()
    ast.archive_list = list(names)
    ast.archive_name = active
    for k, v in over.items():
        setattr(ast, k, v)
    return ast


def test_the_row_shows_the_active_archive_before_any_listing_exists():
    """The listing is built by the orchestrator, so the very first frame has
    none. The row must still name the archive being written to."""
    m = archive_row_model(_ast())
    assert m["names"] == ["default"]
    assert m["index"] == 0


def test_the_row_selects_the_active_archive():
    m = archive_row_model(_ast(TWO, active="dense-trails"))
    assert m["names"][m["index"]] == "dense-trails"


def test_the_row_falls_back_to_the_first_entry_if_the_active_one_is_gone():
    """The folder can vanish outside the app. A stale index would silently
    point the combo at somebody else's archive."""
    m = archive_row_model(_ast(TWO, active="deleted-elsewhere"))
    assert m["index"] == 0


def test_each_label_carries_its_entry_count():
    m = archive_row_model(_ast(TWO))
    assert "1511" in m["labels"][0]


def test_the_summary_describes_the_active_archive_not_the_first():
    m = archive_row_model(_ast(TWO, active="dense-trails"))
    assert "12" in m["summary"] and "1.5" in m["summary"]
    assert m["entries"] == 12


def test_the_summary_says_so_when_nothing_is_loaded_yet():
    m = archive_row_model(_ast())
    assert m["summary"] == "not loaded yet"
    assert m["entries"] == 0


def test_delete_is_disabled_when_there_is_only_one_archive():
    """There must always be something to load."""
    assert archive_row_model(_ast(TWO))["delete_enabled"] is True
    assert archive_row_model(_ast(TWO[:1]))["delete_enabled"] is False
    assert archive_row_model(_ast())["delete_enabled"] is False


def test_a_new_name_is_previewed_sanitised():
    st = new_archive_status(_ast(TWO, new_archive_name="a/b"))
    assert st["safe"] == "ab"
    assert "ab" in st["hint"]


def test_an_unchanged_name_needs_no_preview():
    st = new_archive_status(_ast(TWO, new_archive_name="run-07"))
    assert st["hint"] == ""
    assert st["can_create"] is True


def test_a_duplicate_name_cannot_be_created():
    st = new_archive_status(_ast(TWO, new_archive_name="default"))
    assert st["taken"] is True
    assert st["can_create"] is False
    assert "already exists" in st["hint"]


def test_an_empty_name_cannot_be_created():
    assert new_archive_status(_ast(TWO, new_archive_name="  "))["can_create"] is False
    assert new_archive_status(_ast(TWO))["can_create"] is False


def test_the_explore_tab_renders_the_row_with_a_listing(gui):
    h = Harness(driver=_FakeDriver(), archive=_FakeArchive())
    h.state.archive.archive_list = TWO
    assert frame(h.render_explore_tab) > host_only()


def test_the_explore_tab_renders_the_row_with_no_listing(gui):
    h = Harness(driver=_FakeDriver(), archive=_FakeArchive())
    assert frame(h.render_explore_tab) > host_only()


def test_the_modals_render_when_open(gui):
    """A modal left half-built corrupts the whole ImGui frame, so every window
    in the app disappears at once. Completing the frame proves the stack
    balances."""
    h = Harness(driver=_FakeDriver(), archive=_FakeArchive())
    h.state.archive.archive_list = TWO

    def run():
        imgui.open_popup("New archive")
        h._render_archive_modals(h.state.archive)

    assert frame(run) > host_only()


# ---- adding goals ------------------------------------------------------

def test_the_goal_box_submits_on_enter(gui):
    """enter_returns_true, so a list of goals can be typed straight through
    without reaching for the mouse between every entry."""
    h = Harness(goals=GoalList())
    h.state.archive.new_goal_text = "coral reef"
    assert frame(h.render_explore_tab) > host_only()


def test_the_goal_box_renders_with_the_focus_call_in_the_frame(gui):
    """set_keyboard_focus_here targets the PREVIOUS item at -1. A wrong offset
    or a stray call outside an item is an ImGui stack error, which corrupts the
    whole frame rather than failing here - so rendering at all is the check."""
    h = Harness(goals=GoalList())
    for text in ("", "  ", "a goal"):
        h.state.archive.new_goal_text = text
        assert frame(h.render_explore_tab) > host_only()


def test_an_empty_goal_box_never_requests_an_add(gui):
    """Enter on a blank box must not append an empty goal."""
    h = Harness(goals=GoalList())
    h.state.archive.new_goal_text = "   "
    frame(h.render_explore_tab)
    assert h.state.archive.add_goal_requested is False


# ---- map zoom, pan and hover -------------------------------------------

def _origin_size():
    return imgui.ImVec2(100.0, 200.0), imgui.ImVec2(500.0, 320.0)


def test_home_resets_zoom_and_centre():
    ast = ArchiveState()
    ast.map_zoom, ast.map_center_x, ast.map_center_y = 17.0, 0.1, 0.9
    ArchiveWindowMixin._map_home(ast)
    assert (ast.map_zoom, ast.map_center_x, ast.map_center_y) == (1.0, 0.5, 0.5)


def test_at_home_the_archive_spans_the_canvas():
    """Unit 0 and 1 land on the padded edges, and y is flipped."""
    h = Harness()
    ast = ArchiveState()
    origin, size = _origin_size()
    pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    xs, ys = h._map_to_screen(ast, pts, origin, size)
    pad = ArchiveWindowMixin._MAP_PAD
    assert xs[0] == pytest.approx(origin.x + pad)
    assert xs[1] == pytest.approx(origin.x + size.x - pad)
    assert ys[0] == pytest.approx(origin.y + size.y - pad), "unit y=0 is BOTTOM"
    assert ys[1] == pytest.approx(origin.y + pad)


def test_zoom_keeps_the_centre_put_and_spreads_everything_else():
    h = Harness()
    ast = ArchiveState()
    origin, size = _origin_size()
    pts = np.array([[0.5, 0.5], [0.75, 0.5]], dtype=np.float32)
    x1, _ = h._map_to_screen(ast, pts, origin, size)
    ast.map_zoom = 4.0
    x4, _ = h._map_to_screen(ast, pts, origin, size)
    assert x4[0] == pytest.approx(x1[0]), "the centre must not move"
    assert (x4[1] - x4[0]) == pytest.approx(4.0 * (x1[1] - x1[0]))


def test_panning_the_centre_moves_the_points_the_other_way():
    h = Harness()
    ast = ArchiveState()
    origin, size = _origin_size()
    pts = np.array([[0.5, 0.5]], dtype=np.float32)
    before, _ = h._map_to_screen(ast, pts, origin, size)
    ast.map_center_x = 0.6      # look further right
    after, _ = h._map_to_screen(ast, pts, origin, size)
    assert after[0] < before[0], "the view moved right, so the point moves left"


def test_the_map_renders_zoomed_in(gui):
    """Zooming pushes most points off the canvas; they must be clipped rather
    than drawn over the rest of the tab, and the frame must still balance."""
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.map_zoom = 40.0
    h.state.archive.map_center_x = 0.2
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Refit projection" in labels
    assert "maphome" in invisible_button_labels(
        lambda: h._render_map(h.state.archive, h.archive_obj))


def test_the_hover_card_renders_with_a_thumbnail(gui):
    h = Harness(archive=_populated())

    class _Cache:
        def get(self, name):
            return _FakeTex()

    h.thumb_cache = _Cache()
    assert frame(lambda: h._map_hover_card(h.archive_obj, 0)) > host_only()


def test_the_hover_card_renders_without_a_thumbnail(gui):
    """A missing picture must not leave begin_tooltip unbalanced - an
    unbalanced tooltip stack takes the whole frame down, not just the card."""
    h = Harness(archive=_populated())
    h.thumb_cache = None
    assert frame(lambda: h._map_hover_card(h.archive_obj, 0)) > host_only()


# ---- the browser must not redo O(n) work every frame --------------------
#
# Measured on the real archives before this cache existed: the map cost 9.78 ms
# a frame at 4808 entries and 39.92 ms at the 20000 capacity, of which the
# projection matmul alone was 3.7 ms and 15.9 ms. None of it was new work - the
# archive changes once a generation, roughly every 2.8 s.


def test_the_map_reuses_its_projection_between_frames():
    h = Harness(archive=_populated())
    proj = _spread(h.archive_obj)
    h.archive_projection = proj
    first = h._map_points(h.archive_obj, proj, h.state.archive)
    assert h._map_points(h.archive_obj, proj, h.state.archive) is first


def test_admitting_an_entry_invalidates_the_map():
    """The revision is the whole contract: if it does not move when the
    archive does, the map freezes on stale points and no test would notice."""
    h = Harness(archive=_populated())
    proj = _spread(h.archive_obj)
    first = h._map_points(h.archive_obj, proj, h.state.archive)
    h.archive_obj.revision += 1
    assert h._map_points(h.archive_obj, proj, h.state.archive) is not first


def test_refitting_the_projection_invalidates_the_map():
    h = Harness(archive=_populated())
    proj = _spread(h.archive_obj)
    first = h._map_points(h.archive_obj, proj, h.state.archive)
    proj.fit(h.archive_obj.embeddings)          # bumps proj.version
    assert h._map_points(h.archive_obj, proj, h.state.archive) is not first


def test_the_map_colours_pins_over_their_source():
    h = Harness(archive=_populated())
    proj = _spread(h.archive_obj)
    pts = h._map_points(h.archive_obj, proj, h.state.archive)
    for e, c in zip(h.archive_obj.entries, pts.colors.tolist()):
        want = ArchiveWindowMixin._MAP_COLORS["pin" if e.pinned else e.source]
        assert c == want


def test_the_gallery_reuses_its_sort_between_frames():
    h = Harness(archive=_populated())
    ast = h.state.archive
    first = h._sorted_entries(ast, h.archive_obj)
    assert h._sorted_entries(ast, h.archive_obj) is first


def test_changing_the_sort_mode_invalidates_the_order():
    h = Harness(archive=_populated())
    ast = h.state.archive
    ast.sort_by = "novelty"
    first = h._sorted_entries(ast, h.archive_obj)
    ast.sort_by = "liveness"
    second = h._sorted_entries(ast, h.archive_obj)
    assert second is not first
    assert [e.id for _, e in second] != [e.id for _, e in first]


def test_the_pinned_only_filter_invalidates_the_order():
    h = Harness(archive=_populated())
    ast = h.state.archive
    everything = h._sorted_entries(ast, h.archive_obj)
    ast.pinned_only = True
    only_pins = h._sorted_entries(ast, h.archive_obj)
    assert len(only_pins) < len(everything)
    assert all(e.pinned for _, e in only_pins)


def test_admitting_an_entry_invalidates_the_order():
    h = Harness(archive=_populated())
    ast = h.state.archive
    first = h._sorted_entries(ast, h.archive_obj)
    h.archive_obj.entries.append(h.archive_obj.entries[0])
    h.archive_obj.revision += 1
    assert len(h._sorted_entries(ast, h.archive_obj)) == len(first) + 1


def test_a_very_long_warning_still_renders_its_dismiss_button(gui):
    """These banners carry raw OS error strings. With the button on same_line()
    after the text, a message this long pushed it past the right edge of the
    panel and the warning could not be dismissed at all."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive())
    h.state.archive.warning = (
        "Could not empty 'default': [WinError 5] Access is denied: "
        r"'C:\Users\someone\Documents\Fluoddity\archives\default' -> "
        r"'C:\Users\someone\Documents\Fluoddity\archives\default.cleared-1754800000'")
    assert frame(h.render_explore_tab) > host_only()


SECTIONS = ("Goals", "Rollout", "Exploration", "Admission", "Expeditions")


def _open_all_sections():
    # By storage rather than set_next_item_open(), which only reaches the
    # NEXT item - it would open Goals five times and leave the rest shut,
    # so the test would pass without ever drawing them.
    store = imgui.get_state_storage()
    for name in SECTIONS:
        store.set_int(imgui.get_id(name), 1)


def test_every_settings_section_renders_when_opened(gui):
    """The tab was one unbroken column of about twenty sliders with the browser
    button below all of them. Folding them away is only safe if each section
    still draws - a header whose body raises takes the whole frame with it."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())

    def open_all():
        _open_all_sections()
        h.render_explore_tab()

    closed = frame(h.render_explore_tab)
    assert frame(open_all) > closed, "opening the headers drew nothing extra"


# Every ImGui call whose first argument is a label that becomes the widget's
# ID. text/text_colored/progress_bar/image are absent on purpose: they have no
# ID, so they cannot collide.
_ID_WIDGETS = ("button", "small_button", "checkbox", "combo", "slider_int",
               "slider_float", "input_text", "input_float", "drag_float",
               "collapsing_header", "tree_node_ex", "invisible_button",
               "begin_child", "selectable")


def id_clashes(monkeypatch, draw):
    """-> {id: [label, label]} for every ID two visible widgets shared.

    get_id() is called at the widget's own call site, so it sees the same ID
    stack the widget will: the same label under two different tree nodes,
    child windows or top-level windows is correctly NOT a collision.
    """
    seen = {}
    for fn_name in _ID_WIDGETS:
        real = getattr(imgui, fn_name, None)
        if real is None:                     # widget set differs between builds
            continue

        def wrapper(label, *a, _real=real, **kw):
            seen.setdefault(imgui.get_id(label), []).append(label)
            return _real(label, *a, **kw)

        monkeypatch.setattr(imgui, fn_name, wrapper)

    def render():
        seen.clear()                         # frame() draws twice; keep the last
        draw()

    frame(render)
    return {i: v for i, v in seen.items() if len(v) > 1}


def test_no_two_visible_widgets_in_the_tab_share_an_id(gui, monkeypatch):
    """An ImGui widget's identity IS its label, hashed against the enclosing ID
    stack. Two visible items with the same ID is not cosmetic: ImGui raises its
    "conflicting ID" dialog over the app, and the LOSER stops responding to the
    mouse.

    This shipped. A collapsing header named "Archive" sat in the same window as
    the archive combo, also named "Archive", and the combo - the control that
    picks which archive the search writes into - could not be clicked at all.
    """
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    h.state.archive.warning = "something went wrong"

    def draw():
        _open_all_sections()
        h.render_explore_tab()

    clashes = id_clashes(monkeypatch, draw)
    assert not clashes, f"widgets sharing an ImGui ID: {sorted(clashes.values())}"


def test_no_two_visible_widgets_in_the_browser_share_an_id(gui, monkeypatch):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    h.state.archive.selected_entry_id = 0
    clashes = id_clashes(monkeypatch, h.render_archive_window)
    assert not clashes, f"widgets sharing an ImGui ID: {sorted(clashes.values())}"


# -- map view modes ----------------------------------------------------------
#
# Every mode has to actually draw. An ImGui begin/end imbalance or a None
# dereference in any of them corrupts the whole frame, not just the map.

def _mapped(h):
    return lambda: h._render_map(h.state.archive, h.archive_obj)


@pytest.mark.parametrize("mode", ["source", "novelty", "liveness"])
def test_the_map_renders_in_every_colour_mode(gui, mode):
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.map_color_by = mode
    assert frame(_mapped(h)) > host_only()


@pytest.mark.parametrize("mode", ["points", "density", "points+density"])
def test_the_map_renders_in_every_draw_mode(gui, mode):
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.map_render = mode
    assert frame(_mapped(h)) > host_only()


@pytest.mark.parametrize("mode", ["all", "recent", "novel", "kept", "goal", "source"])
def test_the_map_renders_under_every_filter(gui, mode):
    """Including the ones whose extra control only appears for them, and the
    ones that can select nothing."""
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.map_filter = mode
    frame(_mapped(h))


def test_a_filter_that_matches_nothing_says_so_instead_of_drawing_a_broken_map(gui):
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    ast = h.state.archive
    ast.map_filter = "goal"
    ast.map_filter_goal = "nothing has this goal"
    frame(_mapped(h))          # must not raise, and must not draw a scatter


@pytest.mark.parametrize("filt", ["all", "recent", "novel", "goal", "source"])
def test_the_map_controls_do_not_clash_with_the_windows_other_ids(
        gui, monkeypatch, filt):
    """Two visible items hashing to one ID silently stop one of them
    responding to the mouse - see the ID caveat in CLAUDE.md. Per filter,
    because each one reveals a different extra control."""
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.show_browser = True
    h.state.archive.map_filter = filt
    assert id_clashes(monkeypatch, h.render_archive_window) == {}


def test_selection_follows_the_filter_not_the_row(gui):
    """With a filter on, map row i is not archive entry i. The hover lookup
    resolves through the returned index array; losing it would select a
    plausible-looking wrong entry, which is worse than selecting none."""
    h = Harness(archive=_populated())
    proj = _spread(h.archive_obj)
    ast = h.state.archive
    ast.map_filter = "kept"

    idx = h._map_points(h.archive_obj, proj, ast).idx
    kept = [i for i, e in enumerate(h.archive_obj.entries)
            if e.pinned or e.source in ("summit", "record")]
    assert idx.tolist() == kept
    assert [h.archive_obj.entries[int(i)].pinned for i in idx] == [True] * len(idx)


def test_changing_a_view_option_invalidates_the_cached_points(gui):
    """The cache key covers the view options, or switching colour mode would
    keep drawing the old colours until the next admission."""
    h = Harness(archive=_populated())
    proj = _spread(h.archive_obj)
    ast = h.state.archive

    first = h._map_points(h.archive_obj, proj, ast)
    assert h._map_points(h.archive_obj, proj, ast) is first
    ast.map_color_by = "novelty"
    assert h._map_points(h.archive_obj, proj, ast) is not first


def test_filtering_renormalises_so_what_is_left_fills_the_canvas(gui):
    """Half the value of filtering: fewer points AND they expand, instead of
    staying huddled in whatever corner of the full extent they occupied."""
    h = Harness(archive=_populated(n=40))
    proj = _spread(h.archive_obj)
    ast = h.state.archive
    ast.map_filter = "novel"
    ast.map_novel_pct = 25

    pts = h._map_points(h.archive_obj, proj, ast)
    unit, idx = pts.unit, pts.idx
    assert len(idx) == 10
    assert unit.min() == pytest.approx(0.0, abs=1e-5)
    assert unit.max() == pytest.approx(1.0, abs=1e-5)


def test_the_legend_draws_a_ramp_when_colouring_by_a_number(gui):
    """A ramp with no scale is unreadable. color_button, not text: imgui.text
    renders "##" literally and a block glyph depends on the font."""
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.map_color_by = "novelty"
    with_ramp = frame(_mapped(h))
    h.state.archive.map_color_by = "source"
    assert with_ramp > frame(_mapped(h))


def test_the_legend_says_how_much_is_hidden(gui):
    """A filtered map that does not say so misrepresents the archive."""
    h = Harness(archive=_populated(n=40))
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.map_filter = "novel"
    h.state.archive.map_novel_pct = 25
    seen = []
    real = imgui.text_colored

    def spy(col, text, *a, **kw):
        seen.append(text)
        return real(col, text, *a, **kw)

    imgui.text_colored = spy
    try:
        frame(_mapped(h))
    finally:
        imgui.text_colored = real
    assert any("showing 10 of 40" in s for s in seen), seen


# ---- the density heatmap carries the Colour mode --------------------------

class _RecordingDrawList:
    """Captures add_rect_filled, so what _draw_density chose is assertable."""

    def __init__(self):
        self.rects = []

    def add_rect_filled(self, a, b, col):
        self.rects.append((a.x, a.y, int(col)))


def _density(color_by_value, xs, ys, colors, tvals, cell_size=320.0):
    h = Harness()
    dl = _RecordingDrawList()
    h._draw_density(dl, np.asarray(xs, np.float32), np.asarray(ys, np.float32),
                    imgui.ImVec2(0.0, 0.0),
                    imgui.ImVec2(cell_size, cell_size),
                    np.asarray(colors, np.int64),
                    None if tvals is None else np.asarray(tvals, np.float32))
    return dl.rects


def _rgb(packed):
    return packed & 0xFFFFFF


def _alpha(packed):
    return (packed >> 24) & 255


def test_density_takes_the_regime_colour_when_colouring_by_source():
    """The Colour combo has to mean the same thing in both draw modes. It used
    to be ignored here and every cell was coloured by its count."""
    expedition = ArchiveWindowMixin._MAP_COLORS["expedition"]
    expansion = ArchiveWindowMixin._MAP_COLORS["expansion"]
    # Two points in one cell, three in another far away.
    rects = _density(None, [1.0, 3.0, 200.0, 202.0, 204.0],
                     [1.0, 3.0, 200.0, 202.0, 204.0],
                     [expedition, expedition, expansion, expansion, expansion],
                     None)
    got = {_rgb(c) for _x, _y, c in rects}
    assert got == {_rgb(expedition), _rgb(expansion)}


def test_density_ramps_the_cell_mean_when_colouring_by_a_number():
    from services import map_view as mv

    rects = _density(None, [1.0, 3.0, 200.0], [1.0, 3.0, 200.0],
                     [0, 0, 0], [0.0, 0.0, 1.0])
    by_x = {round(x): c for x, _y, c in rects}
    lo, hi = sorted(by_x)
    assert _rgb(by_x[lo]) == _rgb(int(mv.ramp_colors(np.array([0.0]))[0]))
    assert _rgb(by_x[hi]) == _rgb(int(mv.ramp_colors(np.array([1.0]))[0]))


def test_density_puts_the_count_in_the_opacity():
    """Once colour carries novelty, count is the only thing alpha is left to
    say - and a one-entry cell must not fade to invisible."""
    from services import map_view as mv

    # Eight entries land in the cell at x=0, one in the cell out at x=200.
    positions = [1.0] * 8 + [200.0]
    rects = _density(None, positions, positions, [0] * 9, [0.5] * 9)
    by_x = {round(x): c for x, _y, c in rects}
    crowded, lone = sorted(by_x)
    assert _alpha(by_x[crowded]) > _alpha(by_x[lone])
    assert _alpha(by_x[lone]) >= mv.DENSITY_ALPHA_MIN


def test_density_draws_nothing_when_every_point_is_off_canvas():
    assert _density(None, [-50.0, -60.0], [-50.0, -60.0], [1, 1], None) == []


@pytest.mark.parametrize("mode", ["source", "novelty", "liveness"])
def test_the_density_map_renders_in_every_colour_mode(gui, mode):
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.map_color_by = mode
    h.state.archive.map_render = "points+density"
    assert frame(_mapped(h)) > host_only()


# ---- the canvas owns the mouse wheel --------------------------------------

def test_the_map_canvas_is_a_child_that_absorbs_the_wheel(gui, monkeypatch):
    """ImGui scrolls the hovered window with the same wheel event the zoom
    reads, so the tab scrolled AND the map zoomed. A child with both
    no_scrollbar and no_scroll_with_mouse is what stops the forwarding."""
    seen = {}
    real = imgui.begin_child

    def spy(str_id, size=None, child_flags=0, window_flags=0):
        seen[str_id] = window_flags
        return real(str_id, size, child_flags, window_flags)

    monkeypatch.setattr(imgui, "begin_child", spy)
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    frame(_mapped(h))

    flags = seen.get("map_canvas")
    assert flags is not None, f"no map canvas child; saw {sorted(seen)}"
    assert flags & imgui.WindowFlags_.no_scroll_with_mouse
    # Without NoScrollbar too, ImGui walks up to the parent and scrolls it.
    assert flags & imgui.WindowFlags_.no_scrollbar


# ---- narrow panels ---------------------------------------------------------

def narrow_frame(fn, width=layout.MIN_PANEL_WIDTH, n=3):
    """Render fn in a host window at the narrowest the panels allow."""
    for _ in range(n):
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(width, 720), imgui.Cond_.always)
        imgui.begin("narrow host", True)
        fn()
        imgui.end()
        imgui.render()


def narrow_button_labels(fn):
    seen = []
    real = imgui.button

    def spy(label, *a, **kw):
        seen.append(label)
        return real(label, *a, **kw)

    imgui.button = spy
    try:
        narrow_frame(fn)
    finally:
        imgui.button = real
    return seen


def test_the_explore_tab_keeps_its_buttons_at_the_minimum_width(gui):
    """Rows that overflow do not scroll - ImGui clips them, so a button past
    the right edge cannot be clicked at all."""
    goals = GoalList()
    goals.add("coral reef")
    h = Harness(driver=_TracingDriver(), archive=_populated(), goals=goals)
    labels = narrow_button_labels(h.render_explore_tab)
    for want in ("New##archive", "Empty##archive", "Delete##archive",
                 "Refresh##archive", "Start##explore", "Pause##explore",
                 "Reset Search##explore", "Add Goal"):
        assert want in labels, f"{want} missing at {layout.MIN_PANEL_WIDTH}px"


def test_the_map_keeps_its_buttons_at_the_minimum_width(gui):
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.selected_entry_id = 0
    labels = narrow_button_labels(_mapped(h))
    assert "Refit projection" in labels
    assert "maphome" in invisible_button_labels(_mapped(h))
    assert "Save as config...##map" in labels and "Delete##map" in labels


def test_the_auto_tab_keeps_its_buttons_at_the_minimum_width(gui):
    h = Harness()
    h.auto_service = None
    h.auto_unavailable = ""
    labels = narrow_button_labels(h.render_auto_tournament_tab)
    for want in ("Set", "Reset", "Save best genome...", "Save checkpoint",
                 "Load genome", "Load checkpoint"):
        assert want in labels, f"{want} missing at {layout.MIN_PANEL_WIDTH}px"


# ---- live preview ----------------------------------------------------------

def _browser(h):
    h.state.archive.show_browser = True
    return lambda: h.render_archive_window()


def test_the_browser_offers_a_live_preview_toggle(gui):
    h = Harness(archive=_populated())
    labels = checkbox_labels(_browser(h))
    assert "Live preview" in labels


def test_a_closed_browser_reports_no_hover(gui):
    """preview_entry_id is continuous, not a one-shot - so a browser that is
    not drawn must still say the pointer is over nothing, or the preview it
    started keeps running with no way to end it."""
    h = Harness(archive=_populated())
    h.state.archive.show_browser = False
    h.state.archive.preview_entry_id = 3
    frame(h.render_archive_window)
    assert h.state.archive.preview_entry_id == -1


def test_a_drawn_browser_with_the_pointer_elsewhere_reports_no_hover(gui):
    """The mouse is not over the gallery in a headless frame, so nothing may
    claim the preview."""
    h = Harness(archive=_populated())
    h.state.archive.preview_entry_id = 3
    frame(_browser(h))
    assert h.state.archive.preview_entry_id == -1


def test_the_toggle_survives_an_archive_with_no_entries(gui):
    h = Harness(archive=_FakeArchive())
    h.state.archive.preview_entry_id = 2
    frame(_browser(h))
    assert h.state.archive.preview_entry_id == -1


@pytest.mark.parametrize("mode", ["archive", "auto_tournament"])
def test_the_toggle_renders_disabled_during_tournament_mode(gui, mode):
    h = Harness(archive=_populated())
    getattr(h.state, mode).enabled = True
    assert frame(_browser(h)) > host_only()
    assert "Live preview" in checkbox_labels(_browser(h))


# ---- the archive picker lives in the browser too --------------------------

def test_the_browser_can_switch_archives_on_its_own(gui):
    """Opened from Extras the Explore tab may never be on screen, so the tab
    cannot be the only place with a picker."""
    h = Harness(archive=_populated())
    h.state.archive.archive_list = [
        {"name": "default", "entries": 8, "size_mb": 1.0},
        {"name": "reef", "entries": 40, "size_mb": 5.0},
    ]
    labels = button_labels(_browser(h))
    for want in ("New##archive", "Empty##archive", "Delete##archive",
                 "Refresh##archive"):
        assert want in labels


def test_the_picker_and_the_tab_do_not_collide(gui, monkeypatch):
    """Both windows draw a combo labelled "Archive". A widget's ID includes
    its window, so these are distinct - but only if they really are in
    separate windows."""
    h = Harness(driver=_FakeDriver(), archive=_populated(), goals=GoalList())
    h.state.archive.show_browser = True

    def draw():
        h.render_explore_tab()
        h.render_archive_window()

    assert not id_clashes(monkeypatch, draw)


def test_the_browser_summary_is_not_printed_twice(gui):
    """The browser prints its own entry count, so the row's summary would be
    the same number on two consecutive lines."""
    h = Harness(archive=_populated())
    h.state.archive.archive_list = [
        {"name": "default", "entries": 8, "size_mb": 1.25}]
    seen = []
    real = imgui.text_wrapped

    def spy(text, *a, **kw):
        seen.append(text)
        return real(text, *a, **kw)

    imgui.text_wrapped = spy
    try:
        frame(_browser(h))
    finally:
        imgui.text_wrapped = real
    assert not any("1.2 MB" in t for t in seen)
    assert any("8 / 20000 entries" in t for t in seen)
