"""Actually render the Physics Settings window.

Nothing else did. A window body is where a wrong attribute, a changed
signature or an unbalanced begin/end hides: the static attribute walk cannot
see `self.state.tournament.enabled`, and a mismatched begin_disabled asserts
only once ImGui reaches EndFrame. A test that completes has proved the stack
balances; the checks below cover what it drew.
"""
import pytest

from imgui_bundle import imgui

from state import SimState
from state.multi_load_state import MultiLoadState
from state.preferences_state import PreferencesState
from state.tournament_state import TournamentState
from ui.history_window import HistoryWindowMixin
from ui.physics_window import PhysicsWindowMixin
from ui.slider_widgets import SliderWidgetsMixin


@pytest.fixture(scope="module")
def gui():
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 1400)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    for _ in range(2):
        imgui.new_frame()
        imgui.render()
    yield
    imgui.destroy_context()


class _State:
    def __init__(self):
        self.sim = SimState()
        self.multi_load = MultiLoadState()
        self.tournament = TournamentState()
        self.preferences = PreferencesState()
        # Box Cohorts sits with Number of Cohorts, in a section that is folded
        # by default - ImGui runs no widget inside a closed header.
        for f in vars(self.preferences):
            if f.startswith("physics_group_"):
                setattr(self.preferences, f, True)


class _Keys:
    """Stubbed rather than real: KeybindingManager reads the user's own
    keyboard_controls.json, and the tests must not touch user data."""

    def get_key_display_name(self, action):
        return "K"


class Harness(PhysicsWindowMixin, SliderWidgetsMixin, HistoryWindowMixin):
    def __init__(self):
        self.state = _State()
        self.param_lock_service = None
        self.multi_load_service = None
        self.keybindings = _Keys()
        self.currently_open_project = "test"
        self.force_close_physics_menus = False
        self.physics_menu_bar_has_open_menu = False
        self.save_popup_open = False
        self.physics_window_interaction = False
        self.last_hovered_slider = None


def _draw(harness, frames=2):
    """The first frame of a window only sizes it, so read the second."""
    for _ in range(frames):
        imgui.new_frame()
        try:
            harness.render_physics_settings_window()
        except Exception:
            imgui.end_frame()      # or every later test fails on this frame
            raise
        imgui.render()
    return imgui.get_draw_data()


@pytest.mark.parametrize("tournament_on", [False, True])
def test_the_window_renders_with_and_without_a_tournament(gui, tournament_on):
    """Box Cohorts is disabled under a tournament grid, which is a different
    branch of the body - and a stray begin_disabled asserts inside EndFrame."""
    h = Harness()
    h.state.tournament.enabled = tournament_on
    data = _draw(h)
    assert data.total_vtx_count > 0


def _drawn_checkboxes(monkeypatch, harness):
    seen = []
    real = imgui.checkbox

    def spy(label, value, *a, **k):
        seen.append(label)
        return real(label, value, *a, **k)

    monkeypatch.setattr(imgui, "checkbox", spy)
    _draw(harness)
    return seen


def test_the_box_cohorts_toggle_is_offered(gui, monkeypatch):
    h = Harness()
    assert "Box Cohorts" in _drawn_checkboxes(monkeypatch, h)
    assert h.state.sim.box_cohorts is False, "drawing must not change state"


def test_a_tournament_says_why_the_toggle_is_dead(gui, monkeypatch):
    """A disabled widget withholds its hover, so the reason has to be text."""
    h = Harness()
    h.state.tournament.enabled = True
    said = []
    real = imgui.text_disabled
    monkeypatch.setattr(imgui, "text_disabled",
                        lambda t, *a, **k: (said.append(t), real(t, *a, **k))[1])
    _draw(h)
    assert any("tournament" in t.lower() for t in said)
