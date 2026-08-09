"""Switching archives without restarting.

Order is the whole content of this operation. Flush before closing the store or
the trailing entries are lost - index.jsonl is flushed per entry, but
vectors.npz is only rewritten every 200 admissions. Release the thumbnail cache
before rebuilding or the new archive shows the old one's pictures, because
entry ids restart at 0 in every archive.
"""
import json

import numpy as np
import pytest

from state.archive_state import ArchiveState
from state.preferences_state import PreferencesState


class _Log(list):
    def note(self, what):
        self.append(what)


class _FakeService:
    def __init__(self, log):
        self._log = log

    def pause(self):
        self._log.note("pause")


class _FakeDriver:
    def __init__(self, log):
        self._log = log
        self.archive = None
        self.goals = None

    def end_expedition(self):
        self._log.note("end_expedition")


class _FakeArchive:
    def __init__(self, log):
        self._log = log

    def maybe_flush(self, force=False):
        self._log.note(f"flush(force={force})")


class _FakeGoals:
    def __init__(self, log):
        self._log = log

    def save(self):
        self._log.note("save_goals")


class _FakeStore:
    def __init__(self, log):
        self._log = log

    def close(self):
        self._log.note("close_store")


class _FakeCache:
    def __init__(self, log):
        self._log = log

    def release(self):
        self._log.note("release_thumbs")


class _Bag:
    """Stands in for ui and command_handler: anything can be set on it."""


class _FakeApp:
    def __init__(self, log):
        self.ctx = None
        self.auto_service = _FakeService(log)
        self.imgep_driver = _FakeDriver(log)
        self.archive = _FakeArchive(log)
        self.goal_list = _FakeGoals(log)
        self.archive_store = _FakeStore(log)
        self.thumb_cache = _FakeCache(log)
        self.archive_projection = None
        self._last_projection_size = 0
        self.ui = _Bag()
        self.command_handler = _Bag()

    def _build_archive_set(self, path):
        """The real one. _switch_archive is called unbound, so this has to be
        reachable through self - and the rebuild is what the repoint
        assertions are actually checking."""
        from main import App

        return App._build_archive_set(self, path)

    def _release_archive(self, ui_state):
        """Also the real one: it is the half of the switch that lets go of the
        directory, and the order it does that in is what these tests check."""
        from main import App

        return App._release_archive(self, ui_state)


class _UIState:
    def __init__(self):
        self.archive = ArchiveState()
        self.preferences = PreferencesState()


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A Documents/Fluoddity with two archives in it."""
    import utilities.paths as paths

    monkeypatch.setattr(paths, "get_user_data_dir", lambda: tmp_path)
    root = tmp_path / "archives"
    for name, n in (("default", 3), ("dense-trails", 0)):
        (root / name / "thumbs").mkdir(parents=True)
        with open(root / name / "index.jsonl", "w", encoding="utf-8") as fh:
            for i in range(n):
                fh.write(json.dumps({"id": i}) + "\n")
    return root


def switch(app, name, ui_state):
    from main import App
    return App._switch_archive(app, name, ui_state)


def _admit(arc, n, dim=512):
    from services.archive import Candidate

    for i in range(n):
        e = np.zeros(dim, dtype=np.float32)
        e[i] = 1.0
        arc.consider(
            Candidate(brain=np.zeros((10, 8), dtype=np.float32),
                      physics=np.zeros(8, dtype=np.float32),
                      embedding=e, liveness=1.0, spec="brain"),
            novelty=1.0)


def test_the_switch_happens_in_the_order_that_keeps_data(roots):
    log = _Log()
    app, ui = _FakeApp(log), _UIState()

    assert switch(app, "dense-trails", ui) is True

    assert log == ["pause", "end_expedition", "flush(force=True)", "save_goals",
                   "close_store", "release_thumbs"]


def test_the_switch_repoints_every_holder(roots):
    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    driver = app.imgep_driver

    switch(app, "dense-trails", ui)

    assert app.archive is driver.archive
    assert app.goal_list is driver.goals
    assert app.ui.archive_obj is app.archive
    assert app.ui.archive_goals is app.goal_list
    assert app.ui.archive_projection is app.archive_projection
    assert app.ui.thumb_cache is app.thumb_cache
    assert app.command_handler.archive is app.archive
    assert app.command_handler.goal_list is app.goal_list
    assert app.command_handler.archive_projection is app.archive_projection


def test_the_driver_instance_survives_the_switch(roots):
    """The user's exploration settings live on the driver and are not part of
    the archive."""
    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    driver = app.imgep_driver
    switch(app, "dense-trails", ui)
    assert app.imgep_driver is driver


def test_the_switch_persists_the_choice_and_refreshes_the_listing(roots):
    log = _Log()
    app, ui = _FakeApp(log), _UIState()

    switch(app, "dense-trails", ui)

    assert ui.preferences.archive_name == "dense-trails"
    assert ui.archive.archive_name == "dense-trails"
    assert [a["name"] for a in ui.archive.archive_list] == ["default", "dense-trails"]


def test_the_switch_does_not_resume_the_search(roots):
    """The user pressed a management button, not Start. Resuming for them would
    begin writing into an archive they may only have wanted to look at."""
    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    ui.archive.running = True
    switch(app, "dense-trails", ui)
    assert ui.archive.running is False


def test_the_switch_clears_the_selected_entry(roots):
    """Entry ids restart per archive, so a carried-over selection would name a
    different creature."""
    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    ui.archive.selected_entry_id = 42
    switch(app, "dense-trails", ui)
    assert ui.archive.selected_entry_id == -1


def test_switching_to_a_folder_that_vanished_warns_and_changes_nothing(roots):
    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    before = app.archive

    assert switch(app, "ghost", ui) is False

    assert "no longer on disk" in ui.archive.warning
    assert app.archive is before
    assert log == []


def test_switching_loads_the_target_archives_entries(roots):
    """The point of the whole feature: A's contents come back when you go back
    to A."""
    from services.archive import Archive
    from services.archive_io import ArchiveStore

    store = ArchiveStore(roots / "dense-trails")
    arc = Archive(store=store, liveness_min=0.0)
    _admit(arc, 2)
    arc.maybe_flush(force=True)
    store.close()

    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    switch(app, "dense-trails", ui)

    assert len(app.archive) == 2


def test_an_archive_is_untouched_by_a_round_trip(roots):
    """Build A, switch to B, admit to B, come back to A. A must be exactly as
    it was - this is the entire point of the feature."""
    from services.archive import Archive
    from services.archive_io import ArchiveStore

    store = ArchiveStore(roots / "default")
    a = Archive(store=store, liveness_min=0.0)
    _admit(a, 4)
    a.maybe_flush(force=True)
    store.close()

    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    ui.preferences.archive_name = "default"

    switch(app, "default", ui)
    assert len(app.archive) == 4

    switch(app, "dense-trails", ui)
    assert len(app.archive) == 0
    _admit(app.archive, 2)
    app.archive.maybe_flush(force=True)

    switch(app, "default", ui)
    assert len(app.archive) == 4, "A must not have seen B's entries"

    switch(app, "dense-trails", ui)
    assert len(app.archive) == 2, "B kept what was admitted to it"


def test_goal_lists_do_not_leak_between_archives(roots):
    """A goal list is part of the experiment, not a global preference."""
    log = _Log()
    app, ui = _FakeApp(log), _UIState()

    switch(app, "default", ui)
    app.goal_list.add("coral reef")
    app.goal_list.save()

    switch(app, "dense-trails", ui)
    assert len(app.goal_list.items) == 0

    switch(app, "default", ui)
    assert [i["text"] for i in app.goal_list.items] == ["coral reef"]
