"""Directory-level archive operations.

Which archive is loaded is an experimental variable - novelty is measured
against the archive, so its contents ARE the search's memory. This module is
the only place that knows archives are folders, and it is pure stdlib, so all
of it runs against tmp_path with no GL, no CLIP and no window.
"""
import json

import pytest

from services.archive_library import (
    CLEARED_MARK,
    MAX_NAME,
    list_archives,
    resolve,
    safe_name,
)


def make_archive(root, name, entries=0, thumb_bytes=0):
    """A directory shaped exactly like the one ArchiveStore writes."""
    d = root / name
    (d / "thumbs").mkdir(parents=True)
    with open(d / "index.jsonl", "w", encoding="utf-8") as fh:
        for i in range(entries):
            fh.write(json.dumps({"id": i, "thumb": f"{i:06d}.jpg"}) + "\n")
    if thumb_bytes:
        (d / "thumbs" / "000000.jpg").write_bytes(b"x" * thumb_bytes)
    return d


# ---- safe_name ----------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "../secrets", "..\\secrets", "/etc/passwd", "C:\\Windows",
    "a/b", "a\\b", "with:colon", "star*", "quest?", 'quo"te', "pipe|d",
    "lt<gt>",
])
def test_names_lose_everything_that_could_redirect_a_path(raw):
    out = safe_name(raw)
    assert not any(c in out for c in '\\/:*?"<>|')
    assert not out.startswith(".")
    assert not out.endswith((".", " "))


@pytest.mark.parametrize("raw", ["Dense Trails", "\u0433\u0443\u0441\u0442\u043e\u0439",
                                 "run-07", "a.b.c", "CAPS", "emoji \U0001f9ea"])
def test_ordinary_names_survive_unchanged(raw):
    """Case and unicode are not the problem; separators are."""
    assert safe_name(raw) == raw


@pytest.mark.parametrize("raw", ["", "   ", "...", "..", ".", "/", "\\", "*?",
                                 None, 5, b"bytes"])
def test_a_name_that_sanitises_to_nothing_is_refused(raw):
    assert safe_name(raw) == ""


def test_control_characters_are_dropped():
    assert safe_name("run\x00\x1b07") == "run07"


def test_long_names_are_truncated():
    assert len(safe_name("x" * 200)) == MAX_NAME


def test_truncation_never_leaves_a_trailing_dot():
    """Windows silently trims trailing dots from directory names, so a name
    ending in one would not round-trip: the folder created would not be the
    folder looked up."""
    out = safe_name("x" * (MAX_NAME - 1) + "..")
    assert not out.endswith(".")
    assert out == "x" * (MAX_NAME - 1)


# ---- resolve ------------------------------------------------------------

def test_resolve_puts_the_archive_directly_under_the_root(tmp_path):
    assert resolve(tmp_path, "run-07").parent == tmp_path.resolve()


@pytest.mark.parametrize("raw", ["../escape", "..\\escape", "/abs", "C:\\Windows"])
def test_no_sanitised_name_can_escape_the_root(tmp_path, raw):
    assert tmp_path.resolve() in resolve(tmp_path, raw).parents


def test_an_unusable_name_raises(tmp_path):
    with pytest.raises(ValueError):
        resolve(tmp_path, "   ")


def test_resolve_raises_if_sanitising_ever_stops_working(tmp_path, monkeypatch):
    """Unreachable through the UI, because safe_name runs first. It exists so a
    future caller that skips safe_name fails loudly instead of quietly deleting
    something outside archives/."""
    import services.archive_library as lib

    monkeypatch.setattr(lib, "safe_name", lambda raw: "../escape")
    with pytest.raises(ValueError):
        lib.resolve(tmp_path, "anything")


# ---- list_archives ------------------------------------------------------

def test_an_empty_root_lists_nothing(tmp_path):
    assert list_archives(tmp_path) == []


def test_a_missing_root_lists_nothing_rather_than_raising(tmp_path):
    assert list_archives(tmp_path / "nope") == []


def test_every_archive_is_listed_sorted_by_name(tmp_path):
    make_archive(tmp_path, "zeta")
    make_archive(tmp_path, "alpha")
    assert [a["name"] for a in list_archives(tmp_path)] == ["alpha", "zeta"]


def test_the_entry_count_is_the_index_line_count(tmp_path):
    """index.jsonl is the authority for which entries exist, so its line count
    is honest even when vectors.npz has been quarantined."""
    make_archive(tmp_path, "runs", entries=7)
    assert list_archives(tmp_path)[0]["entries"] == 7


def test_an_archive_with_no_index_yet_counts_zero(tmp_path):
    (tmp_path / "fresh" / "thumbs").mkdir(parents=True)
    assert list_archives(tmp_path)[0]["entries"] == 0


def test_cleared_snapshots_are_not_archives(tmp_path):
    make_archive(tmp_path, "runs", entries=2)
    make_archive(tmp_path, f"runs{CLEARED_MARK}1700000000", entries=2)
    assert [a["name"] for a in list_archives(tmp_path)] == ["runs"]


def test_loose_files_in_the_root_are_ignored(tmp_path):
    make_archive(tmp_path, "runs")
    (tmp_path / "readme.txt").write_text("hi", encoding="utf-8")
    assert [a["name"] for a in list_archives(tmp_path)] == ["runs"]


def test_the_size_counts_the_thumbnails(tmp_path):
    """size_mb is dominated by thumbs/, which is the number the user needs when
    deciding what to delete."""
    make_archive(tmp_path, "big", entries=1, thumb_bytes=2 * 1024 * 1024)
    assert list_archives(tmp_path)[0]["size_mb"] == pytest.approx(2.0, abs=0.1)


from services.archive_library import clear, create, delete  # noqa: E402


# ---- create -------------------------------------------------------------

def test_create_makes_an_archive_shaped_directory(tmp_path):
    res = create(tmp_path, "run-07")
    assert res.ok and res.name == "run-07"
    assert (tmp_path / "run-07" / "thumbs").is_dir()


def test_create_sanitises_the_name_it_reports(tmp_path):
    res = create(tmp_path, "a/b")
    assert res.ok and res.name == "ab"
    assert (tmp_path / "ab").is_dir()


def test_create_refuses_an_unusable_name(tmp_path):
    """Refuse rather than auto-generating 'archive-1'. A name the user did not
    choose is a folder they will not find again."""
    res = create(tmp_path, "   ")
    assert not res.ok
    assert "no usable characters" in res.message


def test_create_refuses_a_duplicate_rather_than_suffixing(tmp_path):
    """Silently creating 'run-2' when the user typed 'run' puts entries
    somewhere they did not ask for."""
    create(tmp_path, "run")
    res = create(tmp_path, "run")
    assert not res.ok
    assert "already exists" in res.message
    assert not (tmp_path / "run-2").exists()


# ---- clear --------------------------------------------------------------

def test_clear_moves_the_contents_aside_and_leaves_an_empty_archive(tmp_path):
    make_archive(tmp_path, "runs", entries=5)
    res = clear(tmp_path, "runs")
    assert res.ok
    assert (tmp_path / "runs" / "thumbs").is_dir()
    assert list_archives(tmp_path)[0]["entries"] == 0


def test_clear_deletes_nothing(tmp_path):
    """The undo: the old contents are one rename away from being back."""
    make_archive(tmp_path, "runs", entries=5)
    clear(tmp_path, "runs")
    aside = [p for p in tmp_path.iterdir() if CLEARED_MARK in p.name]
    assert len(aside) == 1
    assert sum(1 for _ in open(aside[0] / "index.jsonl", encoding="utf-8")) == 5


def test_clearing_twice_in_one_second_does_not_collide(tmp_path):
    """The snapshot name is timestamped to the second; two clicks in the same
    second must not have the second rename fail onto the first."""
    make_archive(tmp_path, "runs", entries=1)
    assert clear(tmp_path, "runs").ok
    (tmp_path / "runs" / "index.jsonl").write_text('{"id": 9}\n', encoding="utf-8")
    assert clear(tmp_path, "runs").ok
    assert len([p for p in tmp_path.iterdir() if CLEARED_MARK in p.name]) == 2


def test_clearing_something_that_is_gone_is_a_warning_not_a_crash(tmp_path):
    """The folder can be deleted outside the app between rendering the dropdown
    and clicking the button."""
    res = clear(tmp_path, "ghost")
    assert not res.ok
    assert "no longer on disk" in res.message


# ---- delete -------------------------------------------------------------

def test_delete_removes_the_whole_tree(tmp_path):
    make_archive(tmp_path, "keep")
    make_archive(tmp_path, "drop", entries=3, thumb_bytes=16)
    assert delete(tmp_path, "drop").ok
    assert not (tmp_path / "drop").exists()
    assert (tmp_path / "keep").is_dir()


def test_delete_refuses_the_last_archive(tmp_path):
    """There must always be something to load."""
    make_archive(tmp_path, "only", entries=3)
    res = delete(tmp_path, "only")
    assert not res.ok
    assert "only archive" in res.message
    assert (tmp_path / "only").is_dir()


def test_deleting_something_that_is_gone_is_a_warning_not_a_crash(tmp_path):
    make_archive(tmp_path, "real")
    make_archive(tmp_path, "other")
    res = delete(tmp_path, "ghost")
    assert not res.ok
    assert "no longer on disk" in res.message


def test_a_cleared_snapshot_does_not_count_as_an_archive_for_delete(tmp_path):
    """Snapshots are not archives, so having one does not make it safe to
    delete the only real archive."""
    make_archive(tmp_path, "only", entries=1)
    make_archive(tmp_path, f"only{CLEARED_MARK}1700000000", entries=1)
    assert not delete(tmp_path, "only").ok


# ---- migration ----------------------------------------------------------

from utilities.paths import (  # noqa: E402
    DEFAULT_ARCHIVE, get_archives_root, migrate_legacy_archive,
)


def test_the_legacy_archive_becomes_default_with_its_contents(tmp_path):
    """A move, not a copy: archives run to hundreds of megabytes of
    thumbnails, and duplicating them to be tidy is worse than moving them."""
    legacy = tmp_path / "archive"
    (legacy / "thumbs").mkdir(parents=True)
    (legacy / "index.jsonl").write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

    root = migrate_legacy_archive(tmp_path)

    assert root == tmp_path / "archives"
    assert not legacy.exists(), "moved, not copied"
    listed = list_archives(root)
    assert [a["name"] for a in listed] == [DEFAULT_ARCHIVE]
    assert listed[0]["entries"] == 2
    assert (root / DEFAULT_ARCHIVE / "thumbs").is_dir()


def test_a_first_launch_with_no_archive_at_all_gets_an_empty_default(tmp_path):
    root = migrate_legacy_archive(tmp_path)
    assert (root / DEFAULT_ARCHIVE / "thumbs").is_dir()
    assert list_archives(root)[0]["entries"] == 0


def test_an_existing_archives_root_is_never_touched(tmp_path):
    """A user who has already migrated and then restores an old backup must not
    have that backup swallowed into a directory that already has contents."""
    make_archive(tmp_path / "archives", "default", entries=4)
    legacy = tmp_path / "archive"
    (legacy / "thumbs").mkdir(parents=True)
    (legacy / "index.jsonl").write_text('{"id": 99}\n', encoding="utf-8")

    root = migrate_legacy_archive(tmp_path)

    assert legacy.is_dir(), "the legacy folder must be left alone"
    assert list_archives(root)[0]["entries"] == 4


def test_migration_is_idempotent(tmp_path):
    (tmp_path / "archive" / "thumbs").mkdir(parents=True)
    (tmp_path / "archive" / "index.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    migrate_legacy_archive(tmp_path)
    migrate_legacy_archive(tmp_path)
    assert [a["name"] for a in list_archives(tmp_path / "archives")] == [DEFAULT_ARCHIVE]


def test_a_failed_move_still_leaves_a_usable_default(tmp_path, monkeypatch):
    """Migration failing must not stop the app from launching."""
    import utilities.paths as paths

    (tmp_path / "archive" / "thumbs").mkdir(parents=True)

    def boom(src, dst):
        raise OSError("locked")

    monkeypatch.setattr(paths.os, "replace", boom)
    root = migrate_legacy_archive(tmp_path)

    assert (root / DEFAULT_ARCHIVE / "thumbs").is_dir()
    assert (tmp_path / "archive").is_dir(), "the user's data is still there"


def test_the_root_hangs_off_the_user_data_directory(tmp_path, monkeypatch):
    import utilities.paths as paths

    monkeypatch.setattr(paths, "get_user_data_dir", lambda: tmp_path)
    assert get_archives_root() == tmp_path / "archives"
