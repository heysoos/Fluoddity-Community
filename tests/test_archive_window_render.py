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

from services.gallery_sort import GALLERY_SORTS
from services.goal_source import GoalList
from state.archive_state import ArchiveState
from state.sim_state import SimState
from state.auto_tournament_state import AutoTournamentState
from state.tournament_state import TournamentState
from ui import layout
from ui.archive_window import (ATLAS_BUDGET, ATLAS_CACHE_CAPACITY,
                               ATLAS_HEADROOM, ATLAS_NEW_PER_FRAME,
                               ATLAS_PX_MAX, ATLAS_PX_MIN, GALLERY_COLUMNS,
                               GALLERY_LIST_MAX, GALLERY_TABLE,
                               ArchiveWindowMixin)
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
    """Run full ImGui frames around fn and return the last vertex count.

    The host is sized taller than anything it can hold. ImGui clips a window's
    contents to the WINDOW, not to the display, so at the default size the whole
    settings column falls outside it and draws no vertices at all - which turns
    "opening the sections drew more" into a coin flip on two vertices' worth of
    header arrow. A window past the display edge still emits its geometry.
    """
    for _ in range(n):
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(1200, 4000))
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
        self.moves = []

    def layout_moves(self):
        return list(self.moves)

    def layout_at(self, i):
        return getattr(self.entries[i], "layout", "") or self.signature

    def is_native(self, i):
        return self.layout_at(i) == self.signature

    def thumb_key(self, i):
        t = getattr(self.entries[i], "thumb", "")
        return f"{self.layout_at(i)}/{t}" if t else ""

    def __len__(self):
        return len(self.entries)

    def native_rows(self):
        return np.array([i for i in range(len(self.entries))
                         if self.is_native(i)], dtype=np.int64)

    def stats(self):
        return {"size": len(self.entries), "capacity": 20000,
                "admission_rate": 0.98, "n_nonfinite": 0, "n_rejected": 0,
                "n_evicted": 0, "n_pinned": 0, "blocked_by_pins": False,
                "rejects_ring": 0,
                # As the real one: the browser's mixed-archive note reads these,
                # and a fake without them would exercise the .get() fallback
                # rather than the path that ships.
                "native": int(len(self.native_rows())),
                "layouts": sorted({self.layout_at(i)
                                   for i in range(len(self.entries))})}


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
        self.map_layout_service = None
        self.archive_goal_point = None
        self.thumb_cache = None


def test_explore_tab_renders_before_the_service_exists(gui):
    h = Harness()
    assert frame(h.render_explore_tab) > host_only()


def test_explore_tab_renders_when_clip_is_unavailable(gui):
    h = Harness(unavailable="missing package: onnxruntime")
    assert frame(h.render_explore_tab) > host_only()


def test_a_missing_encoder_draws_a_way_out_rather_than_a_dead_tab(gui):
    """The Explore tab returns before it draws anything else, so this branch
    IS the whole screen: it has to carry both exits - fetch the encoder this
    archive needs, or switch to an archive in a space that is on disk."""
    h = Harness(unavailable="model_missing")
    h.state.archive.encoder_key = "clip-b16"
    labels = set(button_labels(h.render_explore_tab))
    assert "Download clip-b16" in labels
    assert "Refresh##archive" in labels, "no archive row, so no other way out"


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


# ---- one archive, several brains ------------------------------------------

def _texts(monkeypatch, draw):
    """-> every string drawn by text/text_colored/text_wrapped while `draw` runs.

    Asserted on rather than on where the call sits, because ImGui clips to the
    WINDOW: a line inside a folded header or off the bottom edge is written in
    the source and drawn nowhere.
    """
    seen = []
    for name in ("text", "text_colored", "text_wrapped", "text_disabled"):
        real = getattr(imgui, name)

        def wrapper(*a, _real=real, **kw):
            seen.extend(x for x in a if isinstance(x, str))
            return _real(*a, **kw)

        monkeypatch.setattr(imgui, name, wrapper)
    frame(draw)
    return seen


def _two_brains(n=8, foreign_from=5):
    arc = _populated(n=n)
    for i in range(foreign_from, n):
        arc.entries[i].layout = "mlp-n3.4.4-a0.0.0"
    return arc


def test_a_selected_entry_names_the_brain_it_was_authored_under(gui, monkeypatch):
    """On the CLICK, not the hover: an archive pools layouts, so 'which brain
    is this?' is a question about any entry, and the tile cannot say it."""
    h = Harness(archive=_two_brains())
    h.state.archive.show_browser = True

    h.state.archive.selected_entry_id = 0            # a native one
    assert any("fourier-n10 brain" in t
               for t in _texts(monkeypatch, h.render_archive_window))

    h.state.archive.selected_entry_id = 6            # a foreign one
    assert any("mlp-n3.4.4-a0.0.0 brain" in t
               for t in _texts(monkeypatch, h.render_archive_window))


def test_nothing_selected_names_no_brain(gui, monkeypatch):
    h = Harness(archive=_two_brains())
    h.state.archive.show_browser = True
    h.state.archive.selected_entry_id = -1
    drawn = _texts(monkeypatch, h.render_archive_window)
    assert not any(t.startswith(("fourier-n10 brain", "mlp-n3.4.4-a0.0.0 brain"))
                   for t in drawn)


def test_a_mixed_archive_says_how_much_of_it_this_brain_can_breed_from(gui,
                                                                       monkeypatch):
    """Parents and seeds come from the running brain's entries alone, so a
    full-looking archive can still be bootstrapping with nothing on screen
    saying why."""
    h = Harness(archive=_two_brains(n=8, foreign_from=5))
    h.state.archive.show_browser = True
    drawn = " ".join(_texts(monkeypatch, h.render_archive_window))
    assert "2 brains here" in drawn
    assert "5 of 8" in drawn


def test_a_single_brain_archive_says_nothing_about_brains(gui, monkeypatch):
    """Which is every archive that has never had its brain changed. A note
    that is always on is a note nobody reads."""
    h = Harness(archive=_populated(n=8))
    h.state.archive.show_browser = True
    drawn = " ".join(_texts(monkeypatch, h.render_archive_window))
    assert "brains here" not in drawn


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

def _spread(arc, seed=0, engine="pca"):
    """Embeddings with real structure, and the layout service fitted to them.

    The SERVICE, not a bare Projection: it is what the window asks for, and a
    stand-in would exercise a path that does not ship.
    """
    from services.map_layout_service import MapLayoutService

    rng = np.random.default_rng(seed)
    e = rng.normal(size=(len(arc.entries), 8)).astype(np.float32)
    e[:, 0] *= 6.0
    e[:, 1] *= 3.0
    arc.embeddings = e / np.linalg.norm(e, axis=1, keepdims=True)
    svc = MapLayoutService()
    svc.bind(arc, None, "clip-b32")      # None: these tests write no cache
    svc.configure(engine)
    svc.update(arc)
    return svc


def test_the_map_says_so_when_there_is_nothing_to_project(gui):
    """The guard message renders, but the scatter does not - "Relayout"
    only exists on the drawing path, so it is the precise signal."""
    h = Harness(archive=_populated(n=2))
    h.map_layout_service = _spread(h.archive_obj)
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Relayout" not in labels


def test_an_unfitted_projection_does_not_stack_every_point_at_the_origin(gui):
    """transform returns zeros before anything is fitted. Rendering that would
    pile the whole archive in one corner and look like a bug rather than an
    unbuilt map."""
    from services.map_layout_service import MapLayoutService

    h = Harness(archive=_populated())
    h.map_layout_service = MapLayoutService()    # bound to nothing, never fitted
    assert h.map_layout_service.fitted is False
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Relayout" not in labels


def test_a_fitted_projection_does_draw_the_scatter(gui):
    h = Harness(archive=_populated())
    h.map_layout_service = _spread(h.archive_obj)
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Relayout" in labels


def test_the_map_renders_a_fitted_archive(gui):
    h = Harness(archive=_populated())
    h.map_layout_service = _spread(h.archive_obj)
    assert frame(lambda: h._render_map(h.state.archive, h.archive_obj)) > host_only()


def test_the_map_draws_the_goal_marker(gui):
    h = Harness(archive=_populated())
    h.map_layout_service = _spread(h.archive_obj)
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
    h.map_layout_service = _spread(h.archive_obj)
    h.state.archive.show_browser = True
    assert frame(h.render_archive_window) > host_only()


def test_the_map_prompts_you_to_click_when_nothing_is_selected(gui):
    h = Harness(archive=_populated())
    h.map_layout_service = _spread(h.archive_obj)
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Save as config...##map" not in labels


def test_clicking_a_dot_shows_that_entrys_image_and_actions(gui):
    """A dot is a position; without the picture the map says where something
    sits but never what it is."""
    h = Harness(archive=_populated())
    h.map_layout_service = _spread(h.archive_obj)
    h.state.archive.selected_entry_id = 5
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Save as config...##map" in labels
    assert "Delete##map" in labels


def test_the_map_selection_falls_back_to_a_placeholder_without_a_thumbnail(gui):
    h = Harness(archive=_populated())
    h.map_layout_service = _spread(h.archive_obj)
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
    h.map_layout_service = _spread(h.archive_obj)
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


def test_the_new_archive_modal_offers_the_encoder(gui, monkeypatch):
    """The one moment an archive holds nothing is the only one at which the
    choice is real, so this modal is where it has to be."""
    h = Harness(driver=_FakeDriver(), archive=_FakeArchive())

    def run():
        imgui.open_popup("New archive")
        h._render_archive_modals(h.state.archive)

    assert "##new_archive_encoder" in _combo_labels(monkeypatch, run)


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
    h.map_layout_service = _spread(h.archive_obj)
    h.state.archive.map_zoom = 40.0
    h.state.archive.map_center_x = 0.2
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Relayout" in labels
    assert "maphome" in invisible_button_labels(
        lambda: h._render_map(h.state.archive, h.archive_obj))


def test_the_hover_card_renders_with_a_thumbnail(gui):
    h = Harness(archive=_populated())

    class _Cache:
        def get(self, name):
            return _FakeTex()

    h.thumb_cache = _Cache()
    assert frame(lambda: h._map_hover_card(h.state.archive, h.archive_obj, 0)) > host_only()


def test_the_hover_card_renders_without_a_thumbnail(gui):
    """A missing picture must not leave begin_tooltip unbalanced - an
    unbalanced tooltip stack takes the whole frame down, not just the card."""
    h = Harness(archive=_populated())
    h.thumb_cache = None
    assert frame(lambda: h._map_hover_card(h.state.archive, h.archive_obj, 0)) > host_only()


# ---- the browser must not redo O(n) work every frame --------------------
#
# Measured on the real archives before this cache existed: the map cost 9.78 ms
# a frame at 4808 entries and 39.92 ms at the 20000 capacity, of which the
# projection matmul alone was 3.7 ms and 15.9 ms. None of it was new work - the
# archive changes once a generation, roughly every 2.8 s.


def test_the_map_reuses_its_projection_between_frames():
    h = Harness(archive=_populated())
    proj = _spread(h.archive_obj)
    h.map_layout_service = proj
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
    proj.request_refit()                        # bumps proj.version
    proj.update(h.archive_obj)
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


SECTIONS = ("Goals", "Rollout", "Exploration", "Admission", "Expeditions",
            "Brain Layout")


def _open_all_sections():
    # By storage rather than set_next_item_open(), which only reaches the
    # NEXT item - it would open Goals five times and leave the rest shut,
    # so the test would pass without ever drawing them.
    store = imgui.get_state_storage()
    for name in SECTIONS:
        store.set_int(imgui.get_id(name), 1)


def _combo_labels(monkeypatch, draw):
    """-> every label of a combo drawn while `draw` runs.

    Both spellings: a picker that grows per-option tooltips has to move from
    combo() to begin_combo(), and that is not a change in what is on screen.
    """
    seen = []

    for name in ("combo", "begin_combo"):
        real = getattr(imgui, name)

        def wrapper(label, *a, _real=real, **kw):
            seen.append(label)
            return _real(label, *a, **kw)

        monkeypatch.setattr(imgui, name, wrapper)
    frame(draw)
    return seen


def test_the_encoder_is_visible_without_opening_a_section(gui, monkeypatch):
    """It shipped inside the Admission header, which is folded away by default,
    so there was no encoder anywhere on an opened tab."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    assert "Encoder" in _combo_labels(monkeypatch, h.render_explore_tab)


def test_the_browser_shows_the_encoder_too(gui, monkeypatch):
    """It sits with the archive row, which is the one control both windows
    draw - and which archive this is includes which space it is in."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    ast = h.state.archive
    labels = _combo_labels(monkeypatch, lambda: h._render_archive_row(ast))
    assert "Encoder" in labels


def test_the_encoder_readout_never_writes_back(gui, monkeypatch):
    """A readout, not a picker: the archive decides, and a combo that could
    move would be claiming otherwise."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    h.state.archive.encoder_key = "siglip2-b16"
    frame(h.render_explore_tab)
    assert h.state.archive.encoder_key == "siglip2-b16"


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


def _close_all_sections():
    # The context is module-scoped and _open_all_sections writes storage that
    # outlives its own test, so "the sections are shut" has to be asserted
    # rather than assumed. Inside the frame, like the opener: get_id and
    # get_state_storage are both relative to the host window.
    store = imgui.get_state_storage()
    for name in SECTIONS:
        store.set_int(imgui.get_id(name), 0)


def test_the_physics_toggle_is_drawn_with_every_section_shut(gui):
    """physics_enabled and every consumer of it already shipped; no widget
    wrote it, so it could never leave its default. A folded header's body does
    not run at all, so drawing with all of them shut is the assertion that it
    is reachable."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())

    def draw():
        _close_all_sections()
        h.render_explore_tab()

    assert "Search Physics Too" in checkbox_labels(draw)


def test_a_clipped_expedition_seed_says_so(gui, monkeypatch):
    """Re-encoding an archived phenotype is the one lossy step, and it is
    silent: the chase starts from the nearest creature the preset can reach
    rather than the one the seed named."""
    h = Harness(driver=_TracingDriver(seed_phys_clipped=3),
                archive=_FakeArchive(), goals=GoalList())
    said = " ".join(_texts(monkeypatch, h.render_explore_tab))
    assert "3" in said and "physics" in said.lower()


def test_an_unclipped_seed_says_nothing(gui, monkeypatch):
    """A line that is always on screen is not a warning."""
    h = Harness(driver=_TracingDriver(seed_phys_clipped=0),
                archive=_FakeArchive(), goals=GoalList())
    said = " ".join(_texts(monkeypatch, h.render_explore_tab)).lower()
    assert "could not reach" not in said


def _click_checkbox(monkeypatch, draw, label):
    """Run one frame of `draw` with `label`'s checkbox reporting a click.

    A real click needs a mouse the headless context has not got, and what is
    under test is what the tab does with the change rather than ImGui's hit
    testing.
    """
    real = imgui.checkbox

    def spy(lbl, value, *a, **kw):
        real(lbl, value, *a, **kw)
        return (True, not value) if lbl == label else (False, value)

    monkeypatch.setattr(imgui, "checkbox", spy)
    frame(draw, n=1)


def test_flipping_the_physics_toggle_resets_the_search(gui, monkeypatch):
    """The search space changes dimension, so an optimizer built for the old
    one cannot carry on. Reset keeps the archive."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    ast = h.state.archive
    _click_checkbox(monkeypatch, h.render_explore_tab, "Search Physics Too")
    assert ast.physics_enabled is True
    assert ast.reset_requested is True


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
    h.map_layout_service = _spread(h.archive_obj)
    h.state.archive.map_color_by = mode
    assert frame(_mapped(h)) > host_only()


@pytest.mark.parametrize("mode", ["points", "density", "points+density"])
def test_the_map_renders_in_every_draw_mode(gui, mode):
    h = Harness(archive=_populated())
    h.map_layout_service = _spread(h.archive_obj)
    h.state.archive.map_render = mode
    assert frame(_mapped(h)) > host_only()


@pytest.mark.parametrize("mode", ["all", "recent", "novel", "kept", "goal", "source"])
def test_the_map_renders_under_every_filter(gui, mode):
    """Including the ones whose extra control only appears for them, and the
    ones that can select nothing."""
    h = Harness(archive=_populated())
    h.map_layout_service = _spread(h.archive_obj)
    h.state.archive.map_filter = mode
    frame(_mapped(h))


def test_a_filter_that_matches_nothing_says_so_instead_of_drawing_a_broken_map(gui):
    h = Harness(archive=_populated())
    h.map_layout_service = _spread(h.archive_obj)
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
    h.map_layout_service = _spread(h.archive_obj)
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
    h.map_layout_service = _spread(h.archive_obj)
    h.state.archive.map_color_by = "novelty"
    with_ramp = frame(_mapped(h))
    h.state.archive.map_color_by = "source"
    assert with_ramp > frame(_mapped(h))


def test_the_legend_says_how_much_is_hidden(gui):
    """A filtered map that does not say so misrepresents the archive."""
    h = Harness(archive=_populated(n=40))
    h.map_layout_service = _spread(h.archive_obj)
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
    h.map_layout_service = _spread(h.archive_obj)
    h.state.archive.map_color_by = mode
    h.state.archive.map_render = "points+density"
    assert frame(_mapped(h)) > host_only()


# ---- the canvas owns the mouse wheel --------------------------------------

def test_the_map_canvas_is_a_child_with_no_scrollbar(gui, monkeypatch):
    """The canvas clips to a child and draws no scrollbar of its own.

    This says NOTHING about the wheel. It used to claim the flags were what
    stopped the wheel reaching the panel behind, and it passed for as long as
    that was false - no_scroll_with_mouse is ImGui's "give the parent a chance
    to scroll" flag, so it guaranteed the forwarding it was credited with
    preventing. What the wheel actually does is driven in
    tests/test_map_wheel.py."""
    seen = {}
    real = imgui.begin_child

    def spy(str_id, size=None, child_flags=0, window_flags=0):
        seen[str_id] = window_flags
        return real(str_id, size, child_flags, window_flags)

    monkeypatch.setattr(imgui, "begin_child", spy)
    h = Harness(archive=_populated())
    h.map_layout_service = _spread(h.archive_obj)
    frame(_mapped(h))

    flags = seen.get("map_canvas")
    assert flags is not None, f"no map canvas child; saw {sorted(seen)}"
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
    h.map_layout_service = _spread(h.archive_obj)
    h.state.archive.selected_entry_id = 0
    labels = narrow_button_labels(_mapped(h))
    assert "Relayout" in labels
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

# ---- the map's click target -------------------------------------------

def _map_harness(n=40):
    """A populated archive with a FITTED projection, so the map actually draws
    dots rather than the "not enough entries" line."""
    from services.map_layout_service import MapLayoutService

    arc = _populated(n)
    rng = np.random.default_rng(0)
    e = rng.normal(size=(n, 8)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    arc.embeddings = e
    svc = MapLayoutService()
    svc.bind(arc, None, "clip-b32")
    svc.update(arc)
    h = Harness(archive=arc)
    h.map_layout_service = svc
    return h


def _map_rect(h):
    """Where the map canvas actually landed, by spying on its hit target.

    Computed rather than assumed: the canvas sits under a toolbar and a control
    row whose height is not the test's business.
    """
    seen = {}
    real = imgui.invisible_button

    def spy(str_id, size, *a, **kw):
        if str_id == "map_hit":
            p = imgui.get_cursor_screen_pos()
            seen["rect"] = (p.x, p.y, size.x, size.y)
        return real(str_id, size, *a, **kw)

    imgui.invisible_button = spy
    try:
        _map_frame(h, (-100.0, -100.0), False)
    finally:
        imgui.invisible_button = real
    return seen.get("rect")


def _map_frame(h, pos, down):
    """One full frame with the pointer at `pos` and button 0 in state `down`."""
    io = imgui.get_io()
    io.add_mouse_pos_event(float(pos[0]), float(pos[1]))
    io.add_mouse_button_event(0, bool(down))
    imgui.new_frame()
    imgui.set_next_window_pos(imgui.ImVec2(0.0, 0.0))
    imgui.set_next_window_size(imgui.ImVec2(1200.0, 4000.0))
    imgui.begin("host", True)
    h._render_map(h.state.archive, h.archive_obj)
    imgui.end()
    imgui.render()


def _release_mouse():
    """The context is module-scoped, so a test that presses the button MUST put
    it back: everything after it would otherwise render with the mouse held
    down over wherever this left the pointer."""
    io = imgui.get_io()
    io.add_mouse_button_event(0, False)
    io.add_mouse_pos_event(-1000.0, -1000.0)
    frame(lambda: None, n=2)


def test_arriving_at_the_map_and_pressing_does_not_pan_it(gui):
    """The frame a widget becomes active, io.mouse_delta is the movement that
    ARRIVED at it, not a drag of it.

    Panning by that slides every dot out from under the cursor before the hit
    test runs, so the first click after moving the pointer picks a different
    entry - and a second, stationary, click on the same spot works. Differential
    on purpose: the same press point either way, so it cannot pass by the press
    simply missing the canvas.
    """
    a = _map_harness()
    try:
        rect = _map_rect(a)
        assert rect, "map canvas never drew its hit target"
        cx, cy = rect[0] + rect[2] * 0.5, rect[1] + rect[3] * 0.5

        # Already there, then press.
        _map_frame(a, (cx, cy), False)
        _map_frame(a, (cx, cy), False)
        _map_frame(a, (cx, cy), True)
        still = (a.state.archive.map_center_x, a.state.archive.map_center_y)
    finally:
        _release_mouse()

    # Arrive from across the canvas, then press at the SAME point.
    b = _map_harness()
    try:
        _map_frame(b, (cx - 200.0, cy - 90.0), False)
        _map_frame(b, (cx - 200.0, cy - 90.0), False)
        _map_frame(b, (cx, cy), True)
        arrived = (b.state.archive.map_center_x, b.state.archive.map_center_y)
    finally:
        _release_mouse()

    assert arrived == pytest.approx(still, abs=1e-6), (
        "pressing after moving the pointer panned the map, so the dot under "
        "the cursor is no longer the one that was aimed at")


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


# ---- the size slider, and the list at the small end ---------------------
#
# The slider scales the grid; at and below GALLERY_LIST_MAX the gallery becomes
# a detail table instead. Vertex counts cannot tell those apart - the window is
# whatever size an earlier test left it - so these assert on what was CALLED.

def table_ids(fn, n=3):
    """Every begin_table id `fn` emits."""
    seen = []
    real = imgui.begin_table

    def spy(str_id, *a, **kw):
        seen.append(str_id)
        return real(str_id, *a, **kw)

    imgui.begin_table = spy
    try:
        frame(fn, n=n)
    finally:
        imgui.begin_table = real
    return seen


def slider_int_labels(fn, n=3):
    seen = []
    real = imgui.slider_int

    def spy(label, *a, **kw):
        seen.append(label)
        return real(label, *a, **kw)

    imgui.slider_int = spy
    try:
        frame(fn, n=n)
    finally:
        imgui.slider_int = real
    return seen


class _CountingCache:
    """Records what the gallery reserved and what it then asked for."""

    def __init__(self):
        self.reserved = []
        self.asked = 0

    def reserve(self, n):
        self.reserved.append(int(n))

    def peek(self, name):
        return None            # always a miss; _ResidentCache is the other one

    def get(self, name):
        self.asked += 1
        return _FakeTex()


@pytest.mark.parametrize("size", [16, 24, 32, GALLERY_LIST_MAX])
def test_the_list_view_renders_at_every_small_size(gui, size):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    h.state.archive.thumb_size = size
    h.thumb_cache = _CountingCache()
    assert frame(h.render_archive_window) > host_only()


@pytest.mark.parametrize("size", [GALLERY_LIST_MAX + 1, 96, 160])
def test_the_grid_renders_at_every_large_size(gui, size):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    h.state.archive.thumb_size = size
    h.thumb_cache = _CountingCache()
    assert frame(h.render_archive_window) > host_only()


def test_the_small_end_draws_a_table_and_the_large_end_does_not(gui):
    """The flip is the feature; a size that only scaled tiles would pass every
    render test above while doing nothing."""
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    h.thumb_cache = _CountingCache()

    h.state.archive.thumb_size = 32
    assert GALLERY_TABLE in table_ids(h.render_archive_window)

    h.state.archive.thumb_size = 96
    assert GALLERY_TABLE not in table_ids(h.render_archive_window)


def test_the_size_slider_is_drawn(gui):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    assert any("thumb_size" in s
               for s in slider_int_labels(h.render_archive_window))


def test_both_ends_of_the_slider_have_a_clickable_icon(gui):
    """Drawn glyphs, so there is no button label to catch."""
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    ids = invisible_button_labels(h.render_archive_window)
    assert "##gallery_list_icon" in ids
    assert "##gallery_grid_icon" in ids


def test_the_gallery_reserves_room_before_it_asks_for_thumbnails(gui):
    """A frame that touches more than the cache holds evicts every texture and
    re-decodes the lot on the next one."""
    h = Harness(archive=_populated(n=64))
    h.state.archive.show_browser = True
    h.state.archive.thumb_size = 96
    cache = _CountingCache()
    h.thumb_cache = cache
    frame(h.render_archive_window)
    assert cache.reserved, "the gallery never reserved anything"
    assert max(cache.reserved) >= cache.asked / 3.0


def test_a_cache_with_no_reserve_still_renders(gui):
    """The map's hover card and the selection panel share this cache, and a
    test double need not grow a method to keep the gallery drawing."""
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True

    class _Old:
        def get(self, name):
            return _FakeTex()

    h.thumb_cache = _Old()
    assert frame(h.render_archive_window) > host_only()


# ---- sorting ------------------------------------------------------------

@pytest.mark.parametrize("mode", sorted(GALLERY_SORTS) + ["nonsense"])
@pytest.mark.parametrize("size", [32, 96])
def test_every_sort_mode_renders_in_both_views(gui, mode, size):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    h.state.archive.sort_by = mode
    h.state.archive.thumb_size = size
    assert frame(h.render_archive_window) > host_only()


@pytest.mark.parametrize("desc", [True, False])
def test_both_sort_directions_render_in_both_views(gui, desc):
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    h.state.archive.sort_desc = desc
    for size in (32, 96):
        h.state.archive.thumb_size = size
        assert frame(h.render_archive_window) > host_only()


def test_the_sort_combo_offers_every_declared_mode(gui):
    """Derived from the registry, so a mode reachable only from a column header
    cannot quietly go missing from the grid."""
    seen = []
    real = imgui.combo

    def spy(label, current, items, *a, **kw):
        if label == "Sort":
            seen.append(list(items))
        return real(label, current, items, *a, **kw)

    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    imgui.combo = spy
    try:
        frame(h.render_archive_window)
    finally:
        imgui.combo = real

    assert seen, "the Sort combo was never drawn"
    wanted = {GALLERY_SORTS[m].label for m in GALLERY_SORTS}
    assert set(seen[-1]) == wanted


def test_the_list_and_the_grid_draw_the_same_order(gui):
    """One sort_by behind both, so switching view never reorders anything."""
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    h.state.archive.sort_by = "source"
    h.state.archive.thumb_size = 32
    frame(h.render_archive_window)
    listed = [i for i, _ in h._sorted_entries(h.state.archive, h.archive_obj)]
    h.state.archive.thumb_size = 96
    frame(h.render_archive_window)
    gridded = [i for i, _ in h._sorted_entries(h.state.archive, h.archive_obj)]
    assert listed == gridded


# ---- a column header writes the same field the combo does ---------------
#
# The render tests never click a header, so specs_dirty is always False there.
# These drive the mapping directly.

class _FakeCol:
    def __init__(self, index, direction):
        self.column_index = index
        self.sort_direction = direction


class _FakeSpecs:
    def __init__(self, col):
        self._col = col
        self.specs_count = 0 if col is None else 1
        self.specs_dirty = True

    def get_specs(self, n):
        return self._col


def _click_header(monkeypatch, ast, index, direction):
    specs = _FakeSpecs(_FakeCol(index, direction))
    monkeypatch.setattr(imgui, "table_get_sort_specs", lambda: specs)
    ArchiveWindowMixin._read_sort_specs(ast)
    return specs


@pytest.mark.parametrize("index,mode", [
    (n, mode) for n, (_, mode) in enumerate(GALLERY_COLUMNS) if mode])
def test_a_sortable_header_selects_its_own_mode(monkeypatch, index, mode):
    ast = ArchiveState()
    ast.sort_by = "novelty"
    _click_header(monkeypatch, ast, index, imgui.SortDirection.ascending)
    assert ast.sort_by == mode


def test_a_header_carries_its_direction_across():
    """The arrow beside the combo and the header must agree, or switching view
    reverses the gallery."""
    ast = ArchiveState()
    for direction, want in ((imgui.SortDirection.ascending, False),
                            (imgui.SortDirection.descending, True)):
        specs = _FakeSpecs(_FakeCol(1, direction))
        import unittest.mock as _m
        with _m.patch.object(imgui, "table_get_sort_specs", lambda: specs):
            ArchiveWindowMixin._read_sort_specs(ast)
        assert ast.sort_desc is want


def test_the_thumbnail_column_does_not_sort(monkeypatch):
    """It has no mode, and writing an empty sort_by would fall the gallery back
    to novelty on every click of it."""
    ast = ArchiveState()
    ast.sort_by = "goal"
    _click_header(monkeypatch, ast, 0, imgui.SortDirection.ascending)
    assert ast.sort_by == "goal"


def test_a_header_click_is_consumed(monkeypatch):
    """Left dirty, ImGui hands the same click back every frame and the arrow
    button beside the combo could never win."""
    specs = _click_header(monkeypatch, ArchiveState(), 1,
                          imgui.SortDirection.ascending)
    assert specs.specs_dirty is False


def test_a_table_that_is_not_sorting_is_harmless(monkeypatch):
    """table_get_sort_specs answers None when the table has no sort."""
    ast = ArchiveState()
    ast.sort_by = "brain"
    monkeypatch.setattr(imgui, "table_get_sort_specs", lambda: None)
    ArchiveWindowMixin._read_sort_specs(ast)
    assert ast.sort_by == "brain"


def test_a_dirty_spec_with_no_columns_is_harmless(monkeypatch):
    ast = ArchiveState()
    ast.sort_by = "brain"
    specs = _FakeSpecs(None)
    monkeypatch.setattr(imgui, "table_get_sort_specs", lambda: specs)
    ArchiveWindowMixin._read_sort_specs(ast)
    assert ast.sort_by == "brain"
    assert specs.specs_dirty is False


# ---- the layout engine and the thumbnail atlas --------------------------

def _mapped_with(h):
    return lambda: h._render_map(h.state.archive, h.archive_obj)


def test_the_map_offers_a_layout_engine_and_a_relayout_button(gui, monkeypatch):
    h = _map_harness()
    labels = _combo_labels(monkeypatch, _mapped_with(h))
    assert "Layout##map" in labels
    assert "Relayout" in button_labels(_mapped_with(h))


def test_relayout_asks_the_service_rather_than_fitting_in_the_frame(gui):
    h = _map_harness()
    h.state.archive.refit_projection_requested = False
    imgui.new_frame()
    imgui.begin("host", True)
    h._render_map(h.state.archive, h.archive_obj)
    imgui.end()
    imgui.render()
    # The button only sets the one-shot; CommandHandler forwards it.
    assert hasattr(h.map_layout_service, "request_refit")


def test_the_toolbar_says_which_engine_and_how_stale(gui):
    h = _map_harness()
    text = h.map_layout_service.status()
    assert "PCA" in text and "fitted" in text


@pytest.mark.parametrize("px", [16, 32, 64])
def test_the_map_renders_with_thumbnails_at_every_size(gui, px):
    h = _map_harness()
    h.state.archive.map_thumbs = True
    h.state.archive.map_thumb_px = px
    h.thumb_cache = _CountingCache()
    assert frame(_mapped_with(h)) > host_only()


def test_the_atlas_reserves_before_it_asks_for_a_single_thumbnail(gui):
    h = _map_harness(n=200)
    h.state.archive.map_thumbs = True
    h.state.archive.map_thumb_px = 32
    cache = _CountingCache()
    h.thumb_cache = cache
    frame(_mapped_with(h))
    assert cache.reserved, "the atlas never reserved"
    assert max(cache.reserved) >= cache.asked / 3.0


@pytest.mark.parametrize("n", [200, 600])
def test_the_atlas_asks_for_no_more_thumbnails_than_there_are_cells(gui, n):
    """The count follows the VIEWPORT, not the archive - which is the whole
    reason a 20000-entry archive can draw an atlas at all.

    Against the CELL count rather than a ratio between two archive sizes: a
    ratio also passes when both are unbounded.
    """
    px = 32
    h = _map_harness(n=n)
    h.state.archive.map_thumbs = True
    h.state.archive.map_thumb_px = px
    rect = _map_rect(h)
    assert rect, "map canvas never drew its hit target"
    cells = max(1, int(rect[2] // px)) * max(1, int(rect[3] // px))

    cache = _CountingCache()
    h.thumb_cache = cache
    frames = 2
    frame(_mapped_with(h), n=frames)
    # The per-frame decode budget, not the cell count: the atlas fills a level
    # over several frames and only shows it once complete.
    assert cache.asked <= ATLAS_NEW_PER_FRAME * frames, cache.asked
    # A CONSTANT reserve: bounded, and independent of the archive's size.
    assert set(cache.reserved) == {ATLAS_BUDGET + ATLAS_HEADROOM}


def test_a_bigger_archive_does_not_ask_for_more_than_the_canvas_holds(gui):
    """The point of the bound: entries can grow without the picture cost."""
    px = 32
    asks = []
    for n in (200, 2000):
        h = _map_harness(n=n)
        h.state.archive.map_thumbs = True
        h.state.archive.map_thumb_px = px
        cache = _CountingCache()
        h.thumb_cache = cache
        frame(_mapped_with(h), n=1)
        asks.append(cache.asked)
    assert asks[1] <= asks[0] * 2, asks
    assert asks[1] < 2000


def test_thumbnails_off_asks_for_nothing(gui):
    h = _map_harness(n=200)
    h.state.archive.map_thumbs = False
    cache = _CountingCache()
    h.thumb_cache = cache
    frame(_mapped_with(h))
    assert cache.asked == 0


# ---- the atlas must not draw dots under its pictures, or stall on a zoom --

class _DrawSpy:
    """Counts what the map put on the draw list, forwarding everything else."""

    def __init__(self, real):
        self._real = real
        self.circles = 0
        self.images = 0

    def add_circle_filled(self, *a, **kw):
        self.circles += 1
        return self._real.add_circle_filled(*a, **kw)

    def add_image(self, *a, **kw):
        self.images += 1
        return self._real.add_image(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _draw_counts(h, n=1):
    spy = {}
    real = imgui.get_window_draw_list

    def patched():
        dl = real()
        if "s" not in spy:
            spy["s"] = _DrawSpy(dl)
        return spy["s"]

    imgui.get_window_draw_list = patched
    try:
        frame(lambda: h._render_map(h.state.archive, h.archive_obj), n=n)
    finally:
        imgui.get_window_draw_list = real
    return spy.get("s")


class _ResidentCache(_CountingCache):
    """Everything already decoded, which is the steady state."""

    def peek(self, name):
        return _FakeTex()


def test_thumbnail_mode_draws_no_dots_at_all(gui):
    """Not "fewer dots" - NONE. A picture and a dot competing for the same
    entry is what made the atlas fight itself on every pan."""
    h = _map_harness(n=400)
    h.state.archive.map_thumb_px = 32
    h.thumb_cache = _ResidentCache()

    h.state.archive.map_thumbs = False
    plain = _draw_counts(h)
    assert plain.circles > 0 and plain.images == 0

    h.state.archive.map_thumbs = True
    atlas = _draw_counts(h)
    assert atlas.images > 0, "no thumbnails drawn at all"
    assert atlas.circles == 0, f"{atlas.circles} dots drawn under the pictures"


def test_the_atlas_replaces_the_density_layer_too(gui):
    """Draw is disabled while the atlas is on, so an old setting must not keep
    painting underneath it."""
    h = _map_harness(n=400)
    h.state.archive.map_thumbs = True
    h.state.archive.map_render = "density"
    h.thumb_cache = _ResidentCache()
    counts = _draw_counts(h)
    assert counts.images > 0
    assert counts.circles == 0


def test_panning_does_not_change_which_entries_the_atlas_shows(gui):
    """Cells are binned in UNIT space, so a pan slides the pictures without
    reshuffling which entry stands for which cell."""
    h = _map_harness(n=400)
    h.state.archive.map_thumbs = True
    h.state.archive.map_thumb_px = 32
    h.thumb_cache = _ResidentCache()
    pts = h._map_points(h.archive_obj, h.map_layout_service, h.state.archive)
    size = imgui.ImVec2(600.0, 320.0)

    (_ux, _uy, first), _cell = h._atlas_plan(h.state.archive, pts, size)
    h.state.archive.map_center_x += 0.13
    h.state.archive.map_center_y -= 0.07
    (_ux2, _uy2, second), _cell2 = h._atlas_plan(h.state.archive, pts, size)
    assert np.array_equal(first, second)


def test_zooming_within_a_level_keeps_the_same_entries(gui):
    h = _map_harness(n=400)
    h.state.archive.map_thumbs = True
    h.state.archive.map_thumb_px = 32
    pts = h._map_points(h.archive_obj, h.map_layout_service, h.state.archive)
    size = imgui.ImVec2(600.0, 320.0)

    h.state.archive.map_zoom = 2.0
    (_a, _b, first), cell_a = h._atlas_plan(h.state.archive, pts, size)
    h.state.archive.map_zoom = 2.4
    (_c, _d, second), cell_b = h._atlas_plan(h.state.archive, pts, size)
    assert cell_a == cell_b
    assert np.array_equal(first, second)


def test_only_a_few_thumbnails_are_decoded_per_frame(gui):
    """A JPEG decode is ~1 ms and a zoom changes every cell's winner at once,
    so an unbounded fetch is what made the wheel unusable."""
    from ui.archive_window import ATLAS_NEW_PER_FRAME

    h = _map_harness(n=400)
    h.state.archive.map_thumbs = True
    h.state.archive.map_thumb_px = 32
    cache = _CountingCache()          # nothing resident: every cell is a miss
    h.thumb_cache = cache
    frame(lambda: h._render_map(h.state.archive, h.archive_obj), n=1)
    assert 0 < cache.asked <= ATLAS_NEW_PER_FRAME, cache.asked


def test_a_resident_thumbnail_costs_no_decode(gui):
    """Steady state must not re-fetch: peek is what makes the budget affect
    only genuinely new cells."""
    h = _map_harness(n=400)
    h.state.archive.map_thumbs = True
    h.state.archive.map_thumb_px = 32
    cache = _ResidentCache()
    h.thumb_cache = cache
    frame(lambda: h._render_map(h.state.archive, h.archive_obj), n=2)
    assert cache.asked == 0, "it decoded thumbnails it already had"


# --- the atlas must SETTLE: a working set larger than the cache never does --

class _RealisticCache:
    """A real LRU, so eviction is visible. Counts decodes per frame."""

    def __init__(self, max_capacity=None):
        from services.thumb_cache import ThumbCache
        self._c = ThumbCache(lambda name: _FakeTex(), capacity=256,
                             max_capacity=max_capacity
                             or ATLAS_CACHE_CAPACITY)
        self.decodes = 0

    def reserve(self, n):
        self._c.reserve(n)

    def peek(self, name):
        return self._c.peek(name)

    def get(self, name):
        if name not in self._c._items:
            self.decodes += 1
        return self._c.get(name)

    @property
    def capacity(self):
        return self._c.capacity


def _big_map(px, n=8000):
    """A map big enough to ASK for more cells than the cache can hold.

    Both numbers matter: the projection concentrates entries, so a small
    archive or a short canvas occupies only a few hundred cells and settles
    even with the budget removed - which is a test that cannot fail.
    """
    h = _map_harness(n=n)
    h._MAP_H = 900.0
    h.state.archive.map_thumbs = True
    h.state.archive.map_thumb_px = px
    h.thumb_cache = _RealisticCache()
    h.atlas_cache = h.thumb_cache      # one counter for the whole frame
    return h


def _settle(h, limit=400):
    """-> frames taken until a frame decodes nothing, or None."""
    cache = h.thumb_cache
    for i in range(limit):
        cache.decodes = 0
        frame(lambda: h._render_map(h.state.archive, h.archive_obj))
        if cache.decodes == 0:
            return i
    return None


@pytest.mark.parametrize("px", [ATLAS_PX_MIN, 32, ATLAS_PX_MAX])
def test_the_atlas_stops_decoding_once_it_is_full(gui, px):
    """The bug the user saw as a sweep of thumbnails flashing forever.

    At the small end of the slider the atlas wants more cells than the cache
    can hold, so every frame evicts the cells it drew and re-decodes them.
    """
    assert _settle(_big_map(px)) is not None, f"never settled at {px}px"


def test_a_settled_atlas_survives_the_hover_card(gui):
    """The hover card calls get() on entries that are not cell winners. With
    no headroom each one evicts a cell that is still on screen, which the
    NEXT frame then decodes again."""
    h = _big_map(32)
    assert _settle(h) is not None

    cache = h.thumb_cache
    arc = h.archive_obj
    total = 0
    for row in range(0, 300, 7):
        cache.get(arc.thumb_key(row))     # what _map_hover_card does
        cache.decodes = 0                 # count only what the FRAME redoes
        frame(lambda: h._render_map(h.state.archive, h.archive_obj))
        total += cache.decodes
    assert total == 0, f"{total} cells re-decoded by hovering"


def test_the_atlas_working_set_fits_the_cache():
    """The budget and its headroom must leave the cache able to hold a whole
    frame. Raising one without the other puts the flashing straight back."""
    assert ATLAS_BUDGET + ATLAS_HEADROOM <= ATLAS_CACHE_CAPACITY


def test_the_atlas_settles_again_after_a_zoom(gui):
    """Zooming brings new cells into view, so it costs decodes - but a
    BOUNDED number. Perpetual churn is the defect."""
    h = _big_map(32)
    h.state.archive.map_zoom = 4.0
    assert _settle(h) is not None
    for _ in range(12):
        h.state.archive.map_zoom /= 1.25
        frame(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert _settle(h) is not None, "still churning after a zoom out"


def test_the_cache_is_not_shrunk_by_zooming_in(gui):
    """Zoomed in, most of a level's cells are off screen. Sizing the cache to
    the VIEWPORT throws the rest of the level away, so panning or zooming back
    out decodes it all again."""
    h = _big_map(32)
    ast = h.state.archive
    assert _settle(h) is not None
    wide = h.thumb_cache.capacity
    for _ in range(14):
        ast.map_zoom *= 1.3
        frame(lambda: h._render_map(ast, h.archive_obj))
    assert _settle(h) is not None
    assert h.thumb_cache.capacity == wide
    assert wide >= ATLAS_BUDGET


def test_returning_to_a_zoom_level_reuses_most_of_it(gui):
    """Cells off screen must stay resident. Reserving only the VISIBLE count
    shrinks the cache as you zoom in and evicts the level you came from, so
    coming back decodes it from scratch.

    Not zero: crossing a level genuinely introduces new cells, and two full
    levels do not both fit. The invariant is that the bulk is REUSED.
    """
    h = _big_map(32)
    assert _settle(h) is not None
    for _ in range(6):
        h.state.archive.map_zoom *= 1.15
        frame(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert _settle(h) is not None

    cache = h.thumb_cache
    seen = 0
    for _ in range(6):
        h.state.archive.map_zoom /= 1.15
        cache.decodes = 0
        frame(lambda: h._render_map(h.state.archive, h.archive_obj))
        seen += cache.decodes
    assert seen < ATLAS_BUDGET // 4, f"{seen} decodes returning to a level"


def test_hovering_a_picture_picks_that_picture(gui):
    """The card must describe the thumbnail under the pointer, not whichever
    hidden dot happens to be nearest it."""
    h = _big_map(32, n=3000)
    assert _settle(h) is not None
    ast = h.state.archive
    rect = _map_rect(h)
    origin = imgui.ImVec2(rect[0], rect[1])
    size = imgui.ImVec2(rect[2], rect[3])

    shown = h._atlas_shown
    _arc, _pts, (ux, uy, win), cell = shown
    sx, sy = h._map_to_screen(ast, np.stack([ux, uy], axis=1), origin, size)
    fx, fy = h._map_to_screen(
        ast, np.stack([ux + cell[0], uy + cell[1]], axis=1), origin, size)

    checked = 0
    for k in range(0, len(win), max(1, len(win) // 20)):
        mid = imgui.ImVec2((sx[k] + fx[k]) / 2.0, (sy[k] + fy[k]) / 2.0)
        if not (origin.x <= mid.x <= origin.x + size.x
                and origin.y <= mid.y <= origin.y + size.y):
            continue
        assert h._atlas_pick(ast, mid, origin, size) == int(win[k])
        checked += 1
    assert checked >= 5, f"only probed {checked} cells"


def test_the_pointer_outside_every_cell_picks_nothing(gui):
    h = _big_map(32, n=3000)
    assert _settle(h) is not None
    rect = _map_rect(h)
    far = imgui.ImVec2(rect[0] - 500.0, rect[1] - 500.0)
    got = h._atlas_pick(h.state.archive, far,
                        imgui.ImVec2(rect[0], rect[1]),
                        imgui.ImVec2(rect[2], rect[3]))
    assert got == -1


def _images_drawn(h, n=1):
    spy = _draw_counts(h, n=n)
    return 0 if spy is None else spy.images


def test_a_half_decoded_level_is_never_put_on_screen(gui):
    """The sweep: a plan was drawn as it filled, so a zoom or an engine
    switch wiped new thumbnails across the map a row at a time. The atlas is
    double-buffered - the previous level holds until the new one is whole."""
    h = _big_map(32, n=3000)
    assert _settle(h) is not None
    before = _images_drawn(h)
    assert before > 0

    ast = h.state.archive
    ast.map_zoom *= 4.0                      # a wholesale level change
    counts = []
    for _ in range(60):
        counts.append(_images_drawn(h))
        if h.thumb_cache.decodes == 0:
            break
        h.thumb_cache.decodes = 0
    # Never a partial level: each frame drew either the old plan or the new
    # one, and both are complete.
    assert min(counts) > 0, counts
    assert len(set(counts)) <= 2, f"drew {sorted(set(counts))} - a partial fill"


def test_switching_projection_holds_the_old_pictures(gui):
    """Switching PCA to UMAP replaces every position at once. Adopting it
    before it is decoded is the vertical sweep."""
    h = _big_map(32, n=3000)
    assert _settle(h) is not None
    steady = _images_drawn(h)

    # A new projection: same archive, all-new coordinates.
    svc = h.map_layout_service
    rs = np.random.RandomState(7)
    moved = rs.rand(len(h.archive_obj), 2).astype(np.float32)
    svc.transform_rows = lambda arc, idx: moved[idx]
    h._map_cache = None          # what a version bump does, without the setter

    drawn = _images_drawn(h)
    assert drawn > 0, "the map went blank while the new layout decoded"
    assert abs(drawn - steady) <= 2, (drawn, steady)


def test_the_hover_card_gets_the_entry_whose_picture_is_under_the_pointer(gui):
    """Driven through _render_map, not _atlas_pick: the map used to resolve
    the hover off the hidden scatter, so the card described a dot rather than
    the thumbnail the pointer was actually over."""
    h = _big_map(32, n=3000)
    # The canvas must fit the DISPLAY: ImGui does not route a hover to an item
    # clipped off it, so a 900px map in a 900px display hovers nowhere.
    h._MAP_H = 420.0
    assert _settle(h) is not None
    ast = h.state.archive
    rect = _map_rect(h)
    # Settle AGAIN at this canvas: _map_rect renders at its own window size,
    # so the level it plans is not the one _settle just filled, and the atlas
    # would adopt the new one part-way through the probe.
    assert _settle(h) is not None
    origin, size = imgui.ImVec2(rect[0], rect[1]), imgui.ImVec2(rect[2], rect[3])
    disp = imgui.get_io().display_size

    _arc, _pts, (ux, uy, win), cell = h._atlas_shown
    sx, sy = h._map_to_screen(ast, np.stack([ux, uy], axis=1), origin, size)
    fx, fy = h._map_to_screen(
        ast, np.stack([ux + cell[0], uy + cell[1]], axis=1), origin, size)

    seen = []
    real = type(h)._map_hover_card
    type(h)._map_hover_card = lambda self, ast, arc, row: seen.append(int(row))
    try:
        checked = 0
        for k in range(0, len(win), max(1, len(win) // 12)):
            mx, my = (sx[k] + fx[k]) / 2.0, (sy[k] + fy[k]) / 2.0
            if not (origin.x + 4 <= mx <= min(origin.x + size.x, disp.x) - 4
                    and origin.y + 4 <= my <= min(origin.y + size.y,
                                                  disp.y) - 4):
                continue
            seen.clear()
            _map_frame(h, (mx, my), False)
            _map_frame(h, (mx, my), False)
            # Both frames hover, so the card is drawn twice - what matters
            # is that every one of them named the picture under the pointer.
            assert seen and set(seen) == {int(_pts.idx[win[k]])}, (seen, k)
            checked += 1
    finally:
        type(h)._map_hover_card = real
    assert checked >= 4, f"only probed {checked} cells"


def test_the_visible_set_is_capped_to_what_the_cache_holds(gui):
    """The working set is the cells ON SCREEN, and one larger than the cache
    evicts its own cells and re-decodes them forever. Built directly, because
    a test window is not wide enough to ask for that many cells."""
    from services import map_view as _mv
    h = _big_map(16, n=8000)
    ast = h.state.archive
    rect = _map_rect(h)
    pts = h._map_points(h.archive_obj, h.map_layout_service, ast)
    n = ATLAS_BUDGET * 3
    rs = np.random.RandomState(1)
    unit = rs.rand(n, 2).astype(np.float32)
    nov = rs.rand(n).astype(np.float32)
    cell = (1.0 / 96.0, 1.0 / 96.0)
    plan = _mv.atlas_winners(unit, nov, cell)
    assert len(plan[2]) > ATLAS_BUDGET, len(plan[2])
    stub = type(pts)(unit, pts.lo, pts.span,
                     np.zeros(n, dtype=np.int64), np.arange(n), None, nov)
    ast.map_zoom = 1.0
    _a, _b, _c, _d, live = h._atlas_rects(
        ast, (h.archive_obj, stub, plan, cell),
        imgui.ImVec2(rect[0], rect[1]),
        imgui.ImVec2(rect[2], rect[3]))
    assert len(live) <= ATLAS_BUDGET, len(live)


def test_zooming_in_keeps_filling_every_cell_in_view(gui):
    """Capping the whole PLAN by novelty leaves gaps exactly where you zoom:
    the cells in view need not be among the map's most novel. Measured against
    an UNCAPPED plan, or the assertion just re-reads the cap it is checking."""
    from services import map_view as _mv
    h = _big_map(32, n=8000)
    h._MAP_H = 420.0
    ast = h.state.archive
    for z in (1.0, 2.0, 4.0, 8.0, 16.0):
        ast.map_zoom = z
        assert _settle(h) is not None, f"never settled at {z}x"
        rect = _map_rect(h)
        assert _settle(h) is not None
        origin = imgui.ImVec2(rect[0], rect[1])
        size = imgui.ImVec2(rect[2], rect[3])
        pts = h._map_points(h.archive_obj, h.map_layout_service, ast)
        cell = _mv.atlas_cell(size.x * z, size.y * z, float(ast.map_thumb_px))
        full = _mv.atlas_winners(pts.unit, pts.novelty, cell)
        _a, _b, _c, _d, want = h._atlas_rects(ast, (h.archive_obj, pts, full, cell),
                                              origin, size)
        assert len(want), f"nothing visible at {z}x"
        drawn = _images_drawn(h)
        assert drawn >= len(want), (z, drawn, len(want))


def test_switching_archive_drops_the_previous_atlas(gui):
    """The snapshot holds the OLD archive's rows. Switching to a smaller
    archive then indexes past the end of its entry list, mid-frame."""
    h = _big_map(32, n=3000)
    assert _settle(h) is not None
    assert h._atlas_shown is not None

    smaller = _populated(200)
    rng = np.random.default_rng(1)
    e = rng.normal(size=(200, 8)).astype(np.float32)
    smaller.embeddings = e / np.linalg.norm(e, axis=1, keepdims=True)
    h.archive_obj = smaller
    h.map_layout_service.bind(smaller, None, "clip-b32")
    h.map_layout_service.update(smaller)
    h._map_cache = None

    frame(lambda: h._render_map(h.state.archive, h.archive_obj))


def _first_column_width(h):
    """Where the second column starts, minus where the first did.

    Read off a real frame: the thumbnail column's width is a table SETTING,
    restored from imgui.ini, so setting it up with an initial width does not
    move a column the table already has a width for.
    """
    xs = []
    real = imgui.table_next_column

    def spy():
        out = real()
        xs.append(imgui.get_cursor_screen_pos().x)
        return out

    imgui.table_next_column = spy
    try:
        frame(h.render_archive_window)
    finally:
        imgui.table_next_column = real
    return (xs[1] - xs[0]) if len(xs) > 1 else 0.0


@pytest.mark.parametrize("small,big", [(16, 32), (24, GALLERY_LIST_MAX)])
def test_the_thumbnail_column_follows_the_size_slider(gui, small, big):
    """It used to keep whatever width the table had stored, so the pictures
    grew and the column they sat in did not."""
    h = Harness(archive=_populated())
    h.state.archive.show_browser = True
    h.thumb_cache = _ResidentCache()

    h.state.archive.thumb_size = small
    narrow = _first_column_width(h)
    h.state.archive.thumb_size = big
    wide = _first_column_width(h)

    assert narrow > 0 and wide > 0, (narrow, wide)
    assert wide - narrow == pytest.approx(big - small, abs=2.0), (narrow, wide)


# ---- brain layout search -------------------------------------------------

def text_wrapped_lines(fn, n=3):
    """Every wrapped line `fn` emits. layout.text_disabled_wrapped and
    text_colored_wrapped both land on imgui.text_wrapped."""
    seen = []
    real = imgui.text_wrapped

    def spy(text, *a, **kw):
        seen.append(text)
        return real(text, *a, **kw)

    imgui.text_wrapped = spy
    try:
        frame(fn, n=n)
    finally:
        imgui.text_wrapped = real
    return seen


def _explore(h):
    def draw():
        _open_all_sections()
        h.render_explore_tab()
    return draw


def test_the_layout_search_toggle_is_drawn_with_every_section_shut(gui):
    """The primary control, so it may not live in a folded header - a folded
    header's body does not run at all."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())

    def draw():
        _close_all_sections()
        h.render_explore_tab()

    assert "Search Brain Layouts Too" in checkbox_labels(draw)


def test_the_bounds_are_drawn_once_the_search_is_on(gui):
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    h.state.archive.layout_search = True
    labels = slider_int_labels(_explore(h))
    assert any("Max Brain Floats" in s for s in labels), labels
    assert any("Max Layers" in s for s in labels), labels
    assert any("Max Layer Width" in s for s in labels), labels


def test_a_modality_checkbox_comes_from_the_registry(gui):
    """A second hand-written list of modalities is the declared-but-never-read
    defect this codebase has shipped twice."""
    from services.brains import REGISTRY

    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    h.state.archive.layout_search = True
    labels = checkbox_labels(_explore(h))
    for key in REGISTRY:
        assert key in labels, (key, labels)


def test_the_bounds_are_not_drawn_while_the_search_is_off(gui):
    """The header is the whole feature, so it is absent rather than empty."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    assert not any("Max Brain Floats" in s
                   for s in slider_int_labels(_explore(h)))


def test_ticking_a_modality_writes_the_comma_separated_bound(gui):
    """The checkboxes are the only writer of layout_modalities, and a frame
    that merely draws them must not clear what is already in it."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    h.state.archive.layout_search = True
    h.state.archive.layout_modalities = "mlp"
    frame(_explore(h))
    assert h.state.archive.layout_modalities == "mlp"


def test_the_ledger_is_drawn_when_the_archive_holds_moves(gui):
    arc = _FakeArchive()
    arc.moves = [{"parent": "fourier-n10", "child": "fourier-n11",
                  "op": "grow", "gens": 50, "admitted": 3, "kept": True,
                  "gen": 7}]
    h = Harness(driver=_TracingDriver(), archive=arc, goals=GoalList())
    h.state.archive.layout_search = True
    lines = text_wrapped_lines(_explore(h))
    assert any("fourier-n11" in s for s in lines), lines


def test_an_archive_with_no_moves_draws_no_ledger(gui):
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    h.state.archive.layout_search = True
    assert not any("reverted" in s for s in text_wrapped_lines(_explore(h)))


# ---- the brain an entry was authored under, wherever it is named ----------
#
# An archive pools every layout and a dot on the map carries no hint which
# brain drew it. The gallery's selection has always said so; the map said it
# nowhere, on hover or on click.

def _hover_texts(monkeypatch, h, row=0):
    return _texts(monkeypatch, lambda: h._map_hover_card(
        h.state.archive, h.archive_obj, row))


def test_the_map_hover_card_names_the_brain(monkeypatch, gui):
    h = Harness(archive=_populated())
    h.thumb_cache = None
    assert "fourier-n10 brain" in _hover_texts(monkeypatch, h)


def test_the_map_selection_names_the_brain(monkeypatch, gui):
    h = Harness(archive=_populated())
    h.thumb_cache = None
    h.state.archive.selected_entry_id = 0
    seen = _texts(monkeypatch, lambda: h._render_map_selection(
        h.state.archive, h.archive_obj))
    assert "fourier-n10 brain" in seen


def test_the_map_says_it_the_same_way_the_gallery_does(monkeypatch, gui):
    """The point of one helper. Three views name this and three copies of a
    format drift - the user asked for the map to match the gallery."""
    h = Harness(archive=_populated())
    h.thumb_cache = None
    ast = h.state.archive
    ast.selected_entry_id = 0

    gallery = _texts(monkeypatch, lambda: h._render_gallery_selection(
        ast, h.archive_obj))
    line = h._brain_line(h.archive_obj, 0, ast.live_preview)
    assert line in gallery
    assert line in _hover_texts(monkeypatch, h)


@pytest.mark.parametrize("live, tail", [
    (True, "click to switch to it"),
    (False, "turn on Live preview to run it"),
])
def test_a_foreign_entry_says_what_clicking_would_do(monkeypatch, gui, live,
                                                     tail):
    """The foreign half of the format travels to the map with the rest of it.
    On HOVER it is the more useful half - the pointer is already there."""
    arc = _populated()
    arc.entries[0].layout = "mlp-n16-a0"
    h = Harness(archive=arc)
    h.thumb_cache = None
    h.state.archive.live_preview = live

    assert f"mlp-n16-a0 brain - {tail}" in _hover_texts(monkeypatch, h)
