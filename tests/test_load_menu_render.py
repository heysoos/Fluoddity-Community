"""Actually render the File > Load list with a folder category.

A subfolder of physics_configs is a heading of its own, and the Show picker
restricts the list to one folder - which is how an exhibition machine offers
only the presets chosen for it. Asserted on what a frame DRAWS, because a
folded header or a skipped category draws nothing wherever the call sits.
"""
import pytest
from imgui_bundle import imgui

from state.preferences_state import PreferencesState
from ui.config_browser import ConfigBrowserMixin


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


class _MultiLoad:
    multi_load_enabled = False


class _State:
    def __init__(self):
        self.preferences = PreferencesState()
        self.multi_load = _MultiLoad()


class Harness(ConfigBrowserMixin):
    def __init__(self):
        self.state = _State()
        self.config_files_by_category = {
            "Core": ["core_a"], "Custom": ["custom_b"], "Advanced": [],
            "Exhibition": ["show_c", "show_d"],
        }
        self.cached_configs = {}
        self.delete_confirm_filename = None
        self.delete_confirm_category = None
        self.multi_load_service = None


def _selectables(monkeypatch, h):
    seen = []
    real = imgui.selectable

    def wrapper(label, *a, **kw):
        seen.append(label.split("##")[0])
        return real(label, *a, **kw)

    monkeypatch.setattr(imgui, "selectable", wrapper)
    for _ in range(2):
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(600, 800))
        imgui.begin("host")
        h._render_load_submenu_content(False)
        imgui.end()
        imgui.render()
    return seen


def test_a_folder_category_is_listed_beside_the_built_in_ones(gui, monkeypatch):
    h = Harness()
    drawn = set(_selectables(monkeypatch, h))
    assert {"core_a", "custom_b", "show_c", "show_d"} <= drawn


def test_show_only_lists_that_folder_and_nothing_else(gui, monkeypatch):
    h = Harness()
    h.state.preferences.load_menu_only_category = "Exhibition"
    drawn = set(_selectables(monkeypatch, h))
    assert drawn == {"show_c", "show_d"}


def test_a_remembered_folder_that_has_gone_shows_everything(gui, monkeypatch):
    h = Harness()
    h.state.preferences.load_menu_only_category = "Gone"
    drawn = set(_selectables(monkeypatch, h))
    assert {"core_a", "custom_b"} <= drawn
    assert h.state.preferences.load_menu_only_category == ""
