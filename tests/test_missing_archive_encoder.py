"""An archive whose encoder is not downloaded, and the way back out.

Auto's encoder comes from a combo and is gated on `is_present`, so a missing
one draws the download button. Explore's comes off `encoder.json` and nobody
chose it - which made it the one encoder that answered a missing download with
a raw ONNX error, inside a tab that returns before it draws anything else.
"""
from types import SimpleNamespace

import pytest

import main
from command_handler import CommandHandler
from main import App
from state import UIState


class _App:
    """Only the parts _ensure_archive_service touches."""

    _ensure_archive_service = App._ensure_archive_service

    def __init__(self, encoder, driver=None):
        self.archive_store = SimpleNamespace(encoder=encoder)
        self.imgep_driver = driver
        self.vision_scorer = None
        self.auto_service = SimpleNamespace(driver=None)
        self.archive = None
        self.goal_list = None
        self.tournament_service = None
        self.command_handler = SimpleNamespace(imgep_driver=None)
        self.ui = SimpleNamespace(archive_unavailable="", auto_unavailable="",
                                  archive_driver=None, archive_service=None)
        self.scorer_calls = []
        self.opened = 0

    def _ensure_auto_service(self):
        return True

    def _open_archive(self, ui_state):
        self.opened += 1

    def _ensure_scorer(self, key):
        self.scorer_calls.append(key)
        return True


@pytest.fixture
def present(monkeypatch):
    """Which encoders are on disk, by key."""
    have = {"clip-b32"}
    import tools.fetch_models as fm

    monkeypatch.setattr(fm, "is_present", lambda key: key in have)
    return have


def test_a_missing_encoder_is_the_sentinel_the_ui_can_act_on(present):
    """Not a raw error string: `model_missing` is what draws the button."""
    app = _App("clip-b16")
    assert app._ensure_archive_service(UIState()) is False
    assert app.ui.archive_unavailable == "model_missing"


def test_no_session_is_built_for_weights_that_are_not_there(present):
    """The ONNX error this replaces came from trying anyway."""
    app = _App("clip-b16")
    app._ensure_archive_service(UIState())
    assert app.scorer_calls == []


def test_the_key_reaches_the_ui_so_the_button_can_name_it(present):
    ui_state = UIState()
    _App("siglip2-b16")._ensure_archive_service(ui_state)
    assert ui_state.archive.encoder_key == "siglip2-b16"


def test_an_encoder_that_is_there_is_unaffected(present):
    app = _App("clip-b32")
    assert app._ensure_archive_service(UIState()) is True
    assert app.scorer_calls == ["clip-b32"]
    assert app.ui.archive_unavailable == ""


def test_switching_back_to_a_readable_archive_clears_the_banner(present):
    """The driver already exists on that path, so a clear that only ran where
    the driver is BUILT would leave the dead tab standing for good."""
    app = _App("clip-b16", driver=object())
    app._ensure_archive_service(UIState())
    assert app.ui.archive_unavailable == "model_missing"

    app.archive_store = SimpleNamespace(encoder="clip-b32")
    assert app._ensure_archive_service(UIState()) is True
    assert app.ui.archive_unavailable == ""


# ---- the request the button makes ---------------------------------------

class _Handler(CommandHandler):
    def __init__(self):
        self.auto_service = None
        self.imgep_driver = None
        self.downloaded = []

    def _start_model_download(self, key=None):
        self.downloaded.append(key)


def test_the_download_asks_for_the_archives_own_encoder():
    """Auto's `model_key` is a combo nobody can reach from here."""
    h = _Handler()
    ui_state = UIState()
    ui_state.archive.encoder_key = "clip-l14"
    ui_state.archive.download_model_requested = True

    h._handle_explore(ui_state)
    assert h.downloaded == ["clip-l14"]


def test_it_is_honoured_even_though_the_driver_was_never_built():
    """A missing encoder is exactly WHY there is no driver, so a request
    handled below that early return could never be made."""
    h = _Handler()
    assert h.imgep_driver is None
    ui_state = UIState()
    ui_state.archive.encoder_key = "clip-b16"
    ui_state.archive.download_model_requested = True

    h._handle_explore(ui_state)
    assert h.downloaded == ["clip-b16"]
    assert ui_state.archive.download_model_requested is False, "one-shot"


def test_it_is_not_persisted_with_the_archives_settings():
    """A command replayed on load is what the allowlist exists to stop."""
    from state.archive_state import PERSISTED_FIELDS

    assert "download_model_requested" not in PERSISTED_FIELDS
