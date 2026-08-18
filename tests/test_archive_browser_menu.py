"""Opening the archive browser on its own, from Extras.

Reaching the gallery used to mean Tournament Mode -> Explore -> Open Archive
Browser, which also switched the canvas to a tournament grid and built the CLIP
sessions. Neither is needed to look at saved creatures.
"""
import json

import pytest

from state.archive_state import PERSISTED_FIELDS, ArchiveState
from state.preferences_state import PreferencesState


class _Bag:
    pass


class _FakeApp:
    """Only what _open_archive reaches for.

    Deliberately has NO _ensure_auto_service and NO vision_scorer: touching CLIP
    from this path would raise here, which is the point.
    """

    def __init__(self):
        self.ctx = None
        self.archive = None
        self.goal_list = None
        self.archive_store = None
        self.thumb_cache = None
        self.map_layout_service = None
        self.imgep_driver = None
        self._last_projection_size = 0
        self.ui = _Bag()
        self.command_handler = _Bag()

    def _build_archive_set(self, path):
        from main import App

        return App._build_archive_set(self, path)

    def _archive_path_for(self, name, ast):
        from main import App

        return App._archive_path_for(self, name, ast)

    def _open_archive(self, ui_state):
        """The real one - _open_archive_browser calls it through self, and
        whether it reloads an already-open archive is the point of these
        tests."""
        from main import App

        return App._open_archive(self, ui_state)

    def _load_archive_settings(self, ui_state):
        from main import App

        return App._load_archive_settings(self, ui_state)

    def _restore_archive_layout(self, ui_state):
        from main import App

        return App._restore_archive_layout(self, ui_state)


class _UIState:
    def __init__(self):
        self.archive = ArchiveState()
        self.preferences = PreferencesState()


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Documents/Fluoddity holding one archive."""
    import utilities.paths as paths

    monkeypatch.setattr(paths, "get_user_data_dir", lambda: tmp_path)
    d = tmp_path / "archives" / "default"
    (d / "thumbs").mkdir(parents=True)
    (d / "index.jsonl").write_text("", encoding="utf-8")
    return tmp_path / "archives"


def open_archive(app, ui_state):
    from main import App

    return App._open_archive(app, ui_state)


def open_browser(app, ui_state):
    from main import App

    return App._open_archive_browser(app, ui_state)


# ---- loading ---------------------------------------------------------------

def test_opening_the_browser_loads_an_archive_that_was_never_opened(root):
    app, ui = _FakeApp(), _UIState()
    open_browser(app, ui)
    assert app.archive is not None
    assert ui.archive.show_browser is True


def test_the_browser_needs_no_clip(root):
    """_FakeApp has no _ensure_auto_service and no scorer, so a CLIP-touching
    path would raise. Browsing must not pay for the ONNX sessions, and must
    still work where the optional packages are absent."""
    app, ui = _FakeApp(), _UIState()
    open_browser(app, ui)
    assert app.archive is not None
    assert app.imgep_driver is None


def test_the_gallery_gets_everything_it_draws_from(root):
    app, ui = _FakeApp(), _UIState()
    open_browser(app, ui)
    assert app.ui.archive_obj is app.archive
    assert app.ui.thumb_cache is not None
    assert app.ui.map_layout_service is not None
    # The live preview reads the archive through the command handler.
    assert app.command_handler.archive is app.archive


def test_reopening_does_not_reload(root):
    """load_from_store rescores the whole archive; doing that on every tick of
    a menu checkbox would stall the app."""
    app, ui = _FakeApp(), _UIState()
    open_browser(app, ui)
    first, cache = app.archive, app.thumb_cache
    ui.archive.show_browser = False

    open_browser(app, ui)
    assert app.archive is first
    assert app.thumb_cache is cache
    assert ui.archive.show_browser is True


def test_the_archive_name_reaches_the_ui_and_preferences(root):
    app, ui = _FakeApp(), _UIState()
    open_browser(app, ui)
    assert ui.archive.archive_name == "default"
    assert ui.preferences.archive_name == "default"
    assert [a["name"] for a in ui.archive.archive_list] == ["default"]


# ---- the settings must not close the window that just opened --------------

def test_a_saved_closed_browser_does_not_veto_the_menu(root):
    """show_browser is itself a persisted setting. Loading an archive last
    closed would otherwise shut the window the user just asked for."""
    (root / "default" / "settings.json").write_text(
        json.dumps({"show_browser": False, "alpha": 6.0}), encoding="utf-8")

    app, ui = _FakeApp(), _UIState()
    open_browser(app, ui)
    assert ui.archive.show_browser is True
    assert ui.archive.alpha == 6.0, "the other settings still load"


def test_opening_the_archive_alone_still_honours_the_saved_state(root):
    """_open_archive is the shared half and must NOT force the window open -
    Explore mode calls it too, and there the saved state is what should win."""
    (root / "default" / "settings.json").write_text(
        json.dumps({"show_browser": False}), encoding="utf-8")

    app, ui = _FakeApp(), _UIState()
    ui.archive.show_browser = True
    open_archive(app, ui)
    assert ui.archive.show_browser is False


# ---- the request flag ------------------------------------------------------

def test_the_open_request_is_a_command_not_a_setting():
    """Persisting it would reopen the browser on load, and worse, would keep
    re-firing the loader."""
    assert "open_browser_requested" in vars(ArchiveState())
    assert "open_browser_requested" not in PERSISTED_FIELDS


def test_the_request_defaults_off():
    assert ArchiveState().open_browser_requested is False


# ---- entering Explore mode after browsing ---------------------------------

class _ExploreApp(_FakeApp):
    """_FakeApp plus the pieces the driver needs, so the handoff from
    'browsing' to 'searching' can be exercised."""

    def __init__(self):
        super().__init__()
        self.vision_scorer = object()
        self.tournament_service = object()
        self.auto_service = _Bag()
        self.auto_service.driver = object()
        self.prompt_driver = None
        self.ensure_auto_calls = 0
        self.ensure_scorer_keys = []

    def _ensure_auto_service(self):
        self.ensure_auto_calls += 1
        return True

    def _ensure_scorer(self, model_key):
        """Stubbed for the same reason as _ensure_auto_service: building a real
        encoder from this path is what these tests exist to prevent."""
        self.ensure_scorer_keys.append(model_key)
        return True


def ensure_service(app, ui_state):
    from main import App

    return App._ensure_archive_service(app, ui_state)


@pytest.fixture
def weights_present(monkeypatch):
    """Say the archive's encoder is downloaded.

    Stubbed for the same reason `_ensure_scorer` is: these tests are about the
    handoff from browsing to searching, and without this they pass or fail on
    whether the machine running them happens to hold that encoder.
    """
    import tools.fetch_models as fm

    monkeypatch.setattr(fm, "is_present", lambda key: True)


def test_starting_explore_after_browsing_reuses_the_loaded_archive(
        root, weights_present):
    """load_from_store rescores everything, which is seconds on a full
    archive. Browsing then searching must not pay for it twice."""
    app, ui = _ExploreApp(), _UIState()
    open_browser(app, ui)
    loaded, cache = app.archive, app.thumb_cache

    assert ensure_service(app, ui) is True
    assert app.archive is loaded
    # Searching adopts the archive's own encoder; browsing did not.
    assert app.ensure_scorer_keys == ["clip-b32"]
    assert app.thumb_cache is cache
    assert app.imgep_driver is not None
    assert app.imgep_driver.archive is loaded


def test_the_browser_alone_never_builds_the_scorer(root):
    app, ui = _ExploreApp(), _UIState()
    open_browser(app, ui)
    assert app.ensure_auto_calls == 0
    assert app.imgep_driver is None


def test_explore_still_loads_the_archive_when_the_browser_never_opened(
        root, weights_present):
    app, ui = _ExploreApp(), _UIState()
    assert ensure_service(app, ui) is True
    assert app.archive is not None
    assert ui.archive.archive_name == "default"
