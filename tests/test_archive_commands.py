"""The archive-management one-shots.

These run BEFORE _handle_explore so a switch takes effect on the frame the
button was pressed, and so the settings push that follows lands on the archive
the user just chose.
"""
import pytest

from state.archive_state import ArchiveState


class _UIState:
    def __init__(self):
        self.archive = ArchiveState()


class _Handler:
    """Stands in for CommandHandler, with only the attributes the method under
    test touches."""

    def __init__(self):
        from command_handler import CommandHandler

        self.switched = []
        self.switch_archive = lambda name, ui: (self.switched.append(name), True)[1]
        # The method under test is called unbound, so its own helper has to be
        # reachable through self. It is a staticmethod, so this is the real one.
        self._clear_archive_flags = CommandHandler._clear_archive_flags


def run(handler, ui_state):
    from command_handler import CommandHandler
    return CommandHandler._handle_archive_management(handler, ui_state)


@pytest.fixture
def root(tmp_path, monkeypatch):
    import utilities.paths as paths

    monkeypatch.setattr(paths, "get_user_data_dir", lambda: tmp_path)
    r = tmp_path / "archives"
    (r / "default" / "thumbs").mkdir(parents=True)
    return r


def test_new_creates_the_archive_and_switches_to_it(root, tmp_path):
    h, ui = _Handler(), _UIState()
    ui.archive.new_archive_name = "run-07"
    ui.archive.new_archive_requested = True

    run(h, ui)

    assert (root / "run-07" / "thumbs").is_dir()
    assert h.switched == ["run-07"]


def test_a_refused_name_warns_and_does_not_switch(root, tmp_path):
    h, ui = _Handler(), _UIState()
    ui.archive.new_archive_name = "default"
    ui.archive.new_archive_requested = True

    run(h, ui)

    assert "already exists" in ui.archive.warning
    assert h.switched == []


def test_clear_empties_the_active_archive_and_reloads_it(root, tmp_path):
    (root / "default" / "index.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    h, ui = _Handler(), _UIState()
    ui.archive.clear_archive_requested = True

    run(h, ui)

    assert (root / "default" / "index.jsonl").exists() is False
    assert h.switched == ["default"], "the in-memory archive must be reloaded"


def test_delete_removes_it_and_switches_to_what_is_left(root, tmp_path):
    (root / "keeper" / "thumbs").mkdir(parents=True)
    h, ui = _Handler(), _UIState()
    ui.archive.archive_name = "default"
    ui.archive.delete_archive_requested = True

    run(h, ui)

    assert not (root / "default").exists()
    assert h.switched == ["keeper"]


def test_delete_refuses_the_last_archive(root, tmp_path):
    h, ui = _Handler(), _UIState()
    ui.archive.delete_archive_requested = True

    run(h, ui)

    assert (root / "default").is_dir()
    assert "only archive" in ui.archive.warning
    assert h.switched == []


def test_the_dropdown_selection_switches(root, tmp_path):
    (root / "other" / "thumbs").mkdir(parents=True)
    h, ui = _Handler(), _UIState()
    ui.archive.switch_archive_name = "other"

    run(h, ui)

    assert h.switched == ["other"]


def test_refresh_rebuilds_the_cached_listing(root, tmp_path):
    (root / "added-outside" / "thumbs").mkdir(parents=True)
    h, ui = _Handler(), _UIState()
    ui.archive.refresh_archive_list_requested = True

    run(h, ui)

    assert [a["name"] for a in ui.archive.archive_list] == ["added-outside",
                                                            "default"]


def test_the_flags_are_cleared_after_one_frame(root, tmp_path):
    """Otherwise the modal's Create would fire again every frame forever."""
    h, ui = _Handler(), _UIState()
    ui.archive.new_archive_name = "run-07"
    ui.archive.new_archive_requested = True
    ui.archive.clear_archive_requested = True
    ui.archive.delete_archive_requested = True
    ui.archive.switch_archive_name = "default"
    ui.archive.refresh_archive_list_requested = True

    run(h, ui)

    ast = ui.archive
    assert ast.new_archive_requested is False
    assert ast.clear_archive_requested is False
    assert ast.delete_archive_requested is False
    assert ast.switch_archive_name == ""
    assert ast.refresh_archive_list_requested is False


def test_nothing_happens_without_a_switch_callback(root, tmp_path):
    """Explore mode has never been opened, so App has not wired it up."""
    h, ui = _Handler(), _UIState()
    h.switch_archive = None
    ui.archive.new_archive_name = "run-07"
    ui.archive.new_archive_requested = True

    run(h, ui)

    assert not (root / "run-07").exists()
    assert ui.archive.new_archive_requested is False
