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
