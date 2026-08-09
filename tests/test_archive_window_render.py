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
from state.auto_tournament_state import AutoTournamentState
from state.tournament_state import TournamentState
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
        self.auto_tournament = AutoTournamentState()
        self.tournament = TournamentState()


class _FakeArchive:
    def __init__(self, n=0, dim=8):
        self.entries = []
        self.embeddings = np.zeros((n, dim), dtype=np.float32)

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


ACTIONS = {"Export as config", "Seed a run from here", "Delete"}


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
    # a set: _button_labels runs several frames, so every label repeats
    labels = {x for x in _button_labels(h) if x.startswith("#")}
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
    assert "Export as config##map" not in labels


def test_clicking_a_dot_shows_that_entrys_image_and_actions(gui):
    """A dot is a position; without the picture the map says where something
    sits but never what it is."""
    h = Harness(archive=_populated())
    h.archive_projection = _spread(h.archive_obj)
    h.state.archive.selected_entry_id = 5
    labels = button_labels(lambda: h._render_map(h.state.archive, h.archive_obj))
    assert "Export as config##map" in labels
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
    assert "Home##map" in labels


def test_the_hover_card_renders_with_a_thumbnail(gui):
    h = Harness(archive=_populated())

    class _Cache:
        def get(self, name):
            return _FakeTex()

    h.thumb_cache = _Cache()
    entry = h.archive_obj.entries[0]
    assert frame(lambda: h._map_hover_card(entry)) > host_only()


def test_the_hover_card_renders_without_a_thumbnail(gui):
    """A missing picture must not leave begin_tooltip unbalanced - an
    unbalanced tooltip stack takes the whole frame down, not just the card."""
    h = Harness(archive=_populated())
    h.thumb_cache = None
    entry = h.archive_obj.entries[0]
    assert frame(lambda: h._map_hover_card(entry)) > host_only()
