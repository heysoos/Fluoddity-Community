"""The window layout survives a restart, and the test suite never touches it.

Two halves: WHERE the layout lives (imgui.ini, which used to land beside
whichever folder the app was launched from) and WHICH windows were open, which
ImGui does not record at all.
"""
from __future__ import annotations

import ast
from pathlib import Path

from main import App
from state import UIState
from state.preferences_state import (PreferencesState, load_preferences,
                                     save_preferences)
from ui import ini_path
from utilities import paths

CORE = Path(__file__).resolve().parent.parent / "ui" / "core.py"
MAIN = Path(__file__).resolve().parent.parent / "main.py"

RESTORED_FLAGS = ("show_sidebar", "show_physics_settings_window",
                  "show_video_recording_window", "show_history_window",
                  "show_audio_window", "show_archive_browser")


# --- where the layout lives -------------------------------------------------

def test_the_layout_is_user_data(tmp_path):
    """The app directory is read-only in a packaged build, and running from
    source it is whichever checkout was launched."""
    assert paths.get_imgui_ini_path(tmp_path) == tmp_path / "imgui.ini"


def test_the_real_path_sits_beside_the_other_user_data():
    real = paths.get_imgui_ini_path()
    assert real.is_absolute()
    assert real.parent == paths.get_user_data_dir()


def test_migration_prefers_the_layout_the_user_already_arranged(tmp_path):
    app, user = tmp_path / "app", tmp_path / "user"
    app.mkdir()
    user.mkdir()
    (app / "imgui.ini").write_text("ARRANGED", encoding="utf-8")
    (app / "default_imgui.ini").write_text("BUNDLED", encoding="utf-8")

    paths.migrate_imgui_ini(user, app)
    assert (user / "imgui.ini").read_text(encoding="utf-8") == "ARRANGED"


def test_migration_falls_back_to_the_bundled_default(tmp_path):
    app, user = tmp_path / "app", tmp_path / "user"
    app.mkdir()
    user.mkdir()
    (app / "default_imgui.ini").write_text("BUNDLED", encoding="utf-8")

    paths.migrate_imgui_ini(user, app)
    assert (user / "imgui.ini").read_text(encoding="utf-8") == "BUNDLED"


def test_migration_never_overwrites_a_layout_in_use(tmp_path):
    """It runs at every launch, so a second copy would undo every window the
    user has moved since the first."""
    app, user = tmp_path / "app", tmp_path / "user"
    app.mkdir()
    user.mkdir()
    (app / "imgui.ini").write_text("OLD", encoding="utf-8")
    (user / "imgui.ini").write_text("CURRENT", encoding="utf-8")

    paths.migrate_imgui_ini(user, app)
    assert (user / "imgui.ini").read_text(encoding="utf-8") == "CURRENT"


def test_nothing_to_seed_from_is_not_an_error(tmp_path):
    app, user = tmp_path / "app", tmp_path / "user"
    app.mkdir()
    user.mkdir()
    assert paths.migrate_imgui_ini(user, app) == user / "imgui.ini"
    assert not (user / "imgui.ini").exists()


# --- and the suite keeps its hands off it -----------------------------------

def test_the_suite_never_points_imgui_at_the_real_layout():
    """ImGui reads the file in create_context and writes it in
    destroy_context, so a test run would consume the user's layout and leave
    its own behind. conftest suppresses the install; an object with no
    set_ini_filename is what proves nothing was set.
    """
    assert ini_path.is_suppressed()
    assert ini_path.install(object()) is None


def test_installing_points_at_the_user_data_file(tmp_path, monkeypatch):
    target = tmp_path / "made" / "imgui.ini"
    monkeypatch.setattr(ini_path, "_suppressed", False)
    monkeypatch.setattr(ini_path, "get_imgui_ini_path", lambda: target)
    seen = []
    io = type("IO", (), {"set_ini_filename": staticmethod(seen.append)})()

    assert ini_path.install(io) == str(target)
    assert seen == [str(target)]
    assert target.parent.is_dir(), "ImGui does not create the folder itself"


def test_the_ui_installs_it():
    assert "ini_path.install(io)" in CORE.read_text(encoding="utf-8")


# --- which windows were open ------------------------------------------------

def test_every_restored_flag_is_a_preference():
    fields = PreferencesState.__dataclass_fields__
    for name in RESTORED_FLAGS:
        assert name in fields, f"{name} would not survive a restart"


def test_the_open_windows_round_trip(tmp_path):
    prefs = PreferencesState(
        show_sidebar=False, show_physics_settings_window=False,
        show_video_recording_window=True, show_history_window=True,
        show_audio_window=True, show_archive_browser=True)
    path = tmp_path / "preferences.config"
    save_preferences(prefs, path)

    back = load_preferences(path)
    for name in RESTORED_FLAGS:
        assert getattr(back, name) == getattr(prefs, name), name


def test_the_demo_window_is_not_a_preference():
    """A developer toggle, not part of anyone's layout."""
    assert "show_demo_window" not in PreferencesState.__dataclass_fields__


class _Startup:
    """An App wearing the real restore, with nothing else built yet."""

    _restore_open_windows = App._restore_open_windows

    def __init__(self):
        self.ui = type("UI", (), {"state": UIState()})()


def test_the_audio_panel_reopens():
    app = _Startup()
    app._restore_open_windows(PreferencesState(show_audio_window=True))
    assert app.ui.state.audio.show_window is True


def test_a_panel_that_was_closed_stays_closed():
    app = _Startup()
    app.ui.state.audio.show_window = True
    app._restore_open_windows(PreferencesState(show_audio_window=False))
    assert app.ui.state.audio.show_window is False


def test_the_browser_is_asked_to_open():
    app = _Startup()
    app._restore_open_windows(PreferencesState(show_archive_browser=True))
    assert app.ui.state.archive.show_browser is True
    assert app.ui.state.archive.open_browser_requested is True


def test_a_closed_browser_asks_for_nothing():
    """Opening it loads the archive, which rescores every entry."""
    app = _Startup()
    app._restore_open_windows(PreferencesState(show_archive_browser=False))
    assert app.ui.state.archive.open_browser_requested is False


def test_the_restore_happens_once_and_not_on_the_frame_loop():
    """Per frame it would reload the archive every frame."""
    source = MAIN.read_text(encoding="utf-8")
    assert source.count("self._restore_open_windows(") == 1


def _get_state_body():
    tree = ast.parse(CORE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_state":
            return ast.dump(node)
    raise AssertionError("get_state is gone")


def test_get_state_records_the_two_windows_that_own_their_own_flag():
    """Their visibility lives in the audio and archive states, so nothing else
    copies it into the preferences that get written at exit."""
    body = _get_state_body()
    assert "show_audio_window" in body
    assert "show_archive_browser" in body
