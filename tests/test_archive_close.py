"""Letting go of an archive means letting go of EVERY layout's handle.

An archive holds one ArchiveStore per signature directory, and each keeps its
index.jsonl open for append. Windows refuses to remove a directory with an open
handle, so closing only the running layout's store leaves the rest holding the
folder open - and Clear and Delete fail with WinError 32, which the UI reports
as a warning and the archive stays in the list.
"""
from __future__ import annotations

from pathlib import Path

from services.archive import Archive
from services.archive_io import ArchiveStore
from services.archive_library import delete, list_archives
from services.brains import REGISTRY, default_layout

FOURIER = default_layout()
GABOR = REGISTRY["gabor"].layout_from_settings({})


def a_root(tmp_path) -> Path:
    # delete() refuses to remove the only archive, so there has to be another.
    (tmp_path / "other").mkdir()
    return tmp_path


def an_archive(root):
    store = ArchiveStore(root / "target", layout=FOURIER)
    _admit(store)
    return Archive(store=store, dim=8, layout=FOURIER), store


def _admit(store):
    """One index row, which is what gives a layout a directory AND a handle.

    Both are made on first write now, so a layout that was only visited has
    neither - and a fixture that merely constructs stores would assert that
    every handle is closed while none was ever opened."""
    store.append_index({"id": 0, "novelty": 1.0})
    return store


def test_close_releases_every_layouts_handle(tmp_path):
    arc, _store = an_archive(a_root(tmp_path))
    arc.retarget(GABOR)
    _admit(arc.store)
    assert len(arc.stores) == 2
    arc.close()
    assert all(getattr(s, "_fh", None) is None for s in arc.stores.values())


def test_a_retargeted_archive_can_still_be_deleted(tmp_path):
    """The reported symptom: switching brain and then deleting left the
    archive on disk and in the dropdown."""
    root = a_root(tmp_path)
    arc, _store = an_archive(root)
    arc.retarget(GABOR)
    _admit(arc.store)
    arc.close()
    res = delete(root, "target")
    assert res.ok, res.message
    assert [a["name"] for a in list_archives(root)] == ["other"]


def test_closing_the_running_store_alone_is_not_enough(tmp_path):
    """States the mechanism, so a future release path that closes one handle
    and calls it done fails here rather than in the user's dropdown."""
    root = a_root(tmp_path)
    arc, store = an_archive(root)
    arc.retarget(GABOR)
    _admit(arc.store)
    store.close()
    assert any(getattr(s, "_fh", None) is not None
               for s in arc.stores.values())


def test_close_is_idempotent(tmp_path):
    """The delete path releases, deletes, then switches - which releases the
    same archive a second time."""
    arc, _store = an_archive(a_root(tmp_path))
    arc.retarget(GABOR)
    _admit(arc.store)
    arc.close()
    arc.close()
    assert all(getattr(s, "_fh", None) is None for s in arc.stores.values())


def test_a_store_less_archive_closes_without_complaint(tmp_path):
    Archive(store=None, dim=8).close()


def test_an_archive_loaded_from_disk_closes_every_directory_it_found(tmp_path):
    """load_from_store opens a store per signature directory, which is how a
    multi-layout archive held handles long before retarget existed."""
    root = a_root(tmp_path)
    arc, _store = an_archive(root)
    arc.retarget(GABOR)
    _admit(arc.store)
    arc.close()

    again = Archive(store=ArchiveStore(root / "target", layout=FOURIER),
                    dim=8, layout=FOURIER)
    again.load_from_store()
    assert len(again.stores) >= 2
    again.close()
    assert delete(root, "target").ok


# --- a deleted archive must stay deleted ------------------------------------
#
# Delete rmtree's the folder and then SWITCHES, and the switch saves the
# outgoing archive's settings and goals before letting go of it - the same
# store, now pointing at a directory that is gone. A write that recreates its
# own base therefore puts the archive back, half empty: every entry removed,
# the folder still on disk and still in the dropdown.

def _gone_store(tmp_path):
    """A store whose archive has been deleted out from under it."""
    import shutil

    root = a_root(tmp_path)
    path = root / "doomed"
    store = ArchiveStore(path, FOURIER)
    store.close()
    shutil.rmtree(path)
    assert not path.exists()
    return store, path


def test_the_settings_history_does_not_put_a_deleted_archive_back(tmp_path):
    """The one that bit: append_history mkdir'd its own parent, so the switch
    that follows a delete recreated the folder and wrote a history row into
    it."""
    store, path = _gone_store(tmp_path)
    store.append_history({"v": 0, "cfg": {}})
    assert not path.exists(), (
        "appending a settings-history row recreated the deleted archive")


def test_the_goal_list_does_not_put_a_deleted_archive_back(tmp_path):
    """The second survivor. It writes no directory of its own, so it only
    lands once something else has recreated the base."""
    store, path = _gone_store(tmp_path)
    store.save_goals([{"text": "a smiley face"}])
    assert not path.exists()


def test_a_run_config_does_not_put_a_deleted_archive_back(tmp_path):
    """runs/ is a subdirectory, so this one creates with parents and would
    resurrect the base the same way."""
    store, path = _gone_store(tmp_path)
    store.save_run_config("run-1", "{}")
    assert not path.exists()


def test_the_settings_do_not_put_a_deleted_archive_back(tmp_path):
    store, path = _gone_store(tmp_path)
    store.save_settings({"min_separation": 0.02})
    assert not path.exists()


def test_a_live_archive_still_gets_all_four(tmp_path):
    """The guard is 'do not RESURRECT', never 'do not write'. runs/ must still
    be created on demand, since a fresh archive has none."""
    root = a_root(tmp_path)
    path = root / "live"
    store = ArchiveStore(path, FOURIER)
    store.append_history({"v": 0, "cfg": {}})
    store.save_goals([{"text": "a smiley face"}])
    store.save_settings({"min_separation": 0.02})
    assert store.save_run_config("run-1", "{}") is True
    store.close()

    assert (path / "settings_history.jsonl").is_file()
    assert (path / "goals.json").is_file()
    assert (path / "settings.json").is_file()
    assert (path / "runs" / "run-1.json").is_file()


def test_delete_then_a_final_save_leaves_it_out_of_the_list(tmp_path):
    """The user-visible property, end to end: an archive deleted while it is
    the OPEN one must not come back in the dropdown when the switch that
    follows saves what it was holding."""
    root = a_root(tmp_path)
    path = root / "doomed"
    store = ArchiveStore(path, FOURIER)
    store.save_goals([{"text": "a smiley face"}])
    store.append_history({"v": 0, "cfg": {}})
    store.close()

    assert delete(root, "doomed").ok
    # Exactly what App._switch_archive does next, on the outgoing store.
    store.save_settings({"min_separation": 0.02})
    store.append_history({"v": 1, "cfg": {}})
    store.save_goals([{"text": "a smiley face"}])

    assert "doomed" not in [a["name"] for a in list_archives(root)]
