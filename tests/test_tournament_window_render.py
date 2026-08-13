"""Actually render the tournament window.

Everything behind these widgets was covered by unit tests, but the widget layer
itself never ran outside the real app - which is exactly where a mismatched
begin/end, a wrong enum name or a changed signature hides. ImGui asserts on an
unbalanced stack inside EndFrame, so a test that completes at all has proved
the stack balances; the explicit checks below cover the rest.

No window and no renderer: ImGui only needs a display size and a frame.
"""
import time

import pytest

from imgui_bundle import imgui

from services.auto_tournament_service import AutoTournamentService
from services.run_logger import RunLogger
from services.tournament_service import TournamentService
from state import SimState
from state.archive_state import ArchiveState
from state.auto_tournament_state import AutoTournamentState
from state.tournament_state import TournamentState
from ui.auto_tournament_window import AutoTournamentWindowMixin
from ui.tournament_window import TournamentWindowMixin


@pytest.fixture(scope="module")
def gui():
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 900)
    io.delta_time = 1.0 / 60.0
    # ImGui 1.92 builds its font atlas lazily when the backend claims texture
    # support; without this NewFrame asserts on a missing atlas.
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    # The first frame builds the atlas and emits no geometry; warm it up so
    # whichever test runs first is not the one that pays for it.
    for _ in range(2):
        imgui.new_frame()
        imgui.render()
    yield
    imgui.destroy_context()


class _State:
    def __init__(self):
        self.tournament = TournamentState()
        self.auto_tournament = AutoTournamentState()
        # Stands in for UIState, which gained this block with the Explore tab.
        # The Manual and Auto branches of the tab bar clear it so exactly one
        # mode is enabled; without the field they raise mid-frame and leave the
        # ImGui stack unbalanced, which fails every later test in the file.
        self.archive = ArchiveState()
        self.sim = SimState()


class Harness(TournamentWindowMixin, AutoTournamentWindowMixin):
    """The two mixins under test, with only the attributes they reach for."""

    def __init__(self, auto_service=None, unavailable=""):
        self.state = _State()
        self.state.tournament.enabled = True
        self.tournament_service = (auto_service.tournament if auto_service
                                   else TournamentService(grid=4))
        self.tournament_service.init_population()
        self.auto_service = auto_service
        self.auto_unavailable = unavailable
        self._auto_load_path = ""


class FakeScorer:
    def set_prompt(self, text, distractors=None):
        pass

    def score(self, images):
        import numpy as np

        return np.linspace(0.1, 0.6, len(images)).astype(np.float32)


def frame(fn, n=2):
    """Run full ImGui frames around fn and return the last vertex count.

    Twice, because ImGui emits no geometry for a window on the frame it is
    first created - it is measuring content to auto-size. One frame would make
    the vertex assertions depend on which test ran first.
    """
    for _ in range(n):
        imgui.new_frame()
        imgui.begin("host", True)
        fn()
        imgui.end()
        imgui.render()
    return imgui.get_draw_data().total_vtx_count


def _service(tmp_path, generations=0, grid=4):
    ts = TournamentService(grid=grid)
    ts.init_population()
    svc = AutoTournamentService(ts, scorer=FakeScorer(),
                                logger=RunLogger(root=tmp_path))
    svc.configure(steps_per_gen=20, snapshots_per_gen=1, sim_steps_per_frame=20)
    if generations:
        import numpy as np
        svc.start("glowing coral")
        for _ in range(generations):
            for _ in range(200):
                a = svc.update()
                if a.value == "capture":
                    svc.submit_frames(
                        np.full((ts.tiles, 224, 224, 3), 128, dtype=np.uint8))
                elif a.value == "score":
                    # Scoring is off-thread; the first SCORE frame only
                    # submits. See test_auto_tournament_service.scored().
                    if svc.score_and_tell() is not None:
                        break
                    time.sleep(0.001)
    return svc


# -- the window as a whole ---------------------------------------------------


def test_the_whole_window_renders(gui):
    """Covers the tab bar and the Manual body, including the begin/end that was
    moved when the tab bar was introduced."""
    h = Harness()
    assert frame(h.render_tournament_window) > 0


def test_closing_the_window_disables_it_without_unbalancing_the_stack(gui):
    h = Harness()
    h.state.tournament.enabled = False
    frame(h.render_tournament_window)      # early return path


@pytest.mark.parametrize("grid", [2, 4, 8])
def test_manual_grid_renders_at_every_size(gui, grid):
    h = Harness()
    h.tournament_service = TournamentService(grid=grid)
    h.tournament_service.init_population()
    h.tournament_service.toggle_select(0)
    frame(lambda: h._render_manual_tournament(h.state.tournament,
                                              h.tournament_service.selected))


# -- the auto tab ------------------------------------------------------------


def test_auto_tab_renders_before_any_generation(gui, tmp_path):
    h = Harness(_service(tmp_path))
    assert frame(h.render_auto_tournament_tab) > 0


def test_auto_tab_renders_with_history(gui, tmp_path):
    """Exercises the dual-axis trace and the results table."""
    h = Harness(_service(tmp_path, generations=3))
    assert frame(h.render_auto_tournament_tab) > 0


def test_auto_tab_renders_when_clip_is_missing(gui):
    h = Harness(unavailable="model_missing")
    frame(h.render_auto_tournament_tab)
    h2 = Harness(unavailable="missing package: cmaes")
    frame(h2.render_auto_tournament_tab)


def _combo_labels(monkeypatch, draw):
    """-> every label passed to imgui.combo while `draw` runs.

    Renders for real rather than reading the source: the picker WAS reachable
    on every source-level reading of the tab, just not on any path the running
    app takes.
    """
    seen = []
    real = imgui.combo

    def wrapper(label, *a, **kw):
        seen.append(label)
        return real(label, *a, **kw)

    monkeypatch.setattr(imgui, "combo", wrapper)
    frame(draw)
    return seen


def test_the_encoder_picker_is_on_the_auto_tab_when_the_mode_works(
        gui, tmp_path, monkeypatch):
    """It shipped inside the weights-missing branch, so downloading every
    encoder took the only control that chose between them off the screen."""
    h = Harness(_service(tmp_path))
    assert "Encoder" in _combo_labels(monkeypatch, h.render_auto_tournament_tab)


def test_the_encoder_picker_is_still_there_when_weights_are_missing(gui,
                                                                    monkeypatch):
    """The download button downloads whatever it names, so the two must appear
    together."""
    h = Harness(unavailable="model_missing")
    assert "Encoder" in _combo_labels(monkeypatch, h.render_auto_tournament_tab)


def test_auto_tab_renders_with_tile_mutation_open(gui, tmp_path):
    h = Harness(_service(tmp_path))
    h.state.auto_tournament.tile_mutation_enabled = True
    frame(h.render_auto_tournament_tab)


def test_auto_tab_renders_a_warning_banner(gui, tmp_path):
    h = Harness(_service(tmp_path))
    h.state.auto_tournament.warning = "capture looks wrong"
    frame(h.render_auto_tournament_tab)


# -- goal confirmation -------------------------------------------------------


def test_typing_a_goal_leaves_it_pending_until_submitted(gui, tmp_path):
    """The bug: typed text was indistinguishable from the applied goal."""
    svc = _service(tmp_path)
    h = Harness(svc)
    h.state.auto_tournament.prompt = "glowing coral"
    frame(h.render_auto_tournament_tab)
    assert svc.prompt == "", "nothing should be applied just by typing"

    # What CommandHandler does once the flag is set.
    svc.set_prompt(h.state.auto_tournament.prompt)
    frame(h.render_auto_tournament_tab)
    assert svc.prompt == "glowing coral"


def test_goal_renders_in_every_state(gui, tmp_path):
    svc = _service(tmp_path)
    h = Harness(svc)
    ats = h.state.auto_tournament

    frame(h.render_auto_tournament_tab)             # empty, nothing set
    ats.prompt = "coral"
    frame(h.render_auto_tournament_tab)             # pending
    svc.set_prompt("coral")
    frame(h.render_auto_tournament_tab)             # applied
    ats.prompt = "coral reef"
    frame(h.render_auto_tournament_tab)             # edited again


def test_whitespace_only_edits_do_not_count_as_pending(gui, tmp_path):
    svc = _service(tmp_path)
    svc.set_prompt("coral")
    h = Harness(svc)
    h.state.auto_tournament.prompt = "  coral  "
    frame(h.render_auto_tournament_tab)


# -- reset clears the trace --------------------------------------------------


def test_reset_empties_the_trace_the_ui_reads(gui, tmp_path):
    svc = _service(tmp_path, generations=3)
    h = Harness(svc)
    assert len(svc.logger.history()["fit_best"]) == 3
    frame(h.render_auto_tournament_tab)

    svc.reset()
    assert svc.logger.history()["fit_best"] == []
    frame(h.render_auto_tournament_tab)     # must fall back to the empty state


# -- the save row ------------------------------------------------------------


def button_labels(fn, n=3):
    """Every button label `fn` emits.

    Vertex counts cannot answer this: ImGui culls geometry below the fold, and
    the shared module context means the window size is whatever an earlier
    test left it at. Our Python code runs either way, so labels are honest.
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


def test_save_selected_renders_both_enabled_and_disabled(gui):
    """begin_disabled/end_disabled must balance on both branches: an
    unbalanced pair corrupts the whole frame, not just this button."""
    h = Harness()
    h.tournament_service.selected = set()
    assert "Save Selected..." in button_labels(h.render_tournament_window)

    h.tournament_service.selected = {2, 5}
    assert "Save Selected..." in button_labels(h.render_tournament_window)


def test_the_save_notice_is_shown_and_dismissable(gui):
    """A save lands in a folder that is not on screen, so the confirmation has
    to be in the window."""
    h = Harness()
    h.state.tournament.notice = "Saved reef.json to your configs folder."
    labels = button_labels(h.render_tournament_window)
    assert any(lbl.startswith("Dismiss") for lbl in labels)


def test_the_dismiss_button_does_not_collide_with_the_explore_tab(gui):
    """Two visible items with one ImGui id silently stop one of them
    responding to the mouse - see the ID caveat in CLAUDE.md."""
    from state.archive_state import ArchiveState
    from ui.notices import OK, render_banner

    a, b = TournamentState(), ArchiveState()
    a.notice = b.notice = "x"
    ids = []
    for obj, scope in ((a, "tournament"), (b, "explore")):
        imgui.new_frame()
        imgui.begin("host", True)
        render_banner(obj, "notice", OK, scope=scope)
        ids.append(imgui.get_id(f"Dismiss##{scope}notice"))
        imgui.end()
        imgui.render()
    assert ids[0] != ids[1]


# ---- a closed window must leave its sub-modes off -------------------------

def test_closing_the_window_turns_off_auto_and_explore():
    """Only the tab that is drawn sets its own mode. With the window shut
    nothing runs, so a mode left enabled keeps the app applying the tournament
    overrides - forced grid, square tiles, no motion blur - to what the user
    sees as an ordinary single simulation."""
    h = Harness()
    h.state.tournament.enabled = False
    h.state.auto_tournament.enabled = True
    h.state.archive.enabled = True

    h.render_tournament_window()          # returns before begin(); no frame

    assert not h.state.auto_tournament.enabled
    assert not h.state.archive.enabled


def test_a_closed_window_keeps_them_off_on_later_frames():
    h = Harness()
    h.state.tournament.enabled = False
    for _ in range(3):
        h.render_tournament_window()
    assert not h.state.auto_tournament.enabled
    assert not h.state.archive.enabled


def test_the_open_window_still_enables_the_tab_that_is_showing(gui):
    """The close path must not fight the tab bar."""
    h = Harness()
    h.state.tournament.enabled = True
    frame(h.render_tournament_window)
    assert h.state.tournament.enabled
