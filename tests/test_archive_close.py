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
    return Archive(store=store, dim=8, layout=FOURIER), store


def test_close_releases_every_layouts_handle(tmp_path):
    arc, _store = an_archive(a_root(tmp_path))
    arc.retarget(GABOR)
    assert len(arc.stores) == 2
    arc.close()
    assert all(getattr(s, "_fh", None) is None for s in arc.stores.values())


def test_a_retargeted_archive_can_still_be_deleted(tmp_path):
    """The reported symptom: switching brain and then deleting left the
    archive on disk and in the dropdown."""
    root = a_root(tmp_path)
    arc, _store = an_archive(root)
    arc.retarget(GABOR)
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
    store.close()
    assert any(getattr(s, "_fh", None) is not None
               for s in arc.stores.values())


def test_close_is_idempotent(tmp_path):
    """The delete path releases, deletes, then switches - which releases the
    same archive a second time."""
    arc, _store = an_archive(a_root(tmp_path))
    arc.retarget(GABOR)
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
    arc.close()

    again = Archive(store=ArchiveStore(root / "target", layout=FOURIER),
                    dim=8, layout=FOURIER)
    again.load_from_store()
    assert len(again.stores) >= 2
    again.close()
    assert delete(root, "target").ok
