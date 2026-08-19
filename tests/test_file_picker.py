"""The Open-file dialog is optional, non-blocking, and never the only way in.

It runs in a separate process, so a caller polls it: a frame that waited on it
would freeze the sim behind the window for as long as someone was browsing.
"""
from __future__ import annotations

from pathlib import Path

from services import field_sources, file_picker


class _FakeHandle:
    """The backend's contract: ready(timeout) then result()."""

    def __init__(self, ready_after, value):
        self._left = ready_after
        self._value = value

    def ready(self, _timeout):
        if self._left > 0:
            self._left -= 1
            return False
        return True

    def result(self):
        return self._value


def _pending(handle):
    pick = file_picker.PendingPick.__new__(file_picker.PendingPick)
    pick._handle = handle
    return pick


def test_a_dialog_still_browsing_reports_nothing_yet():
    """None means keep going; "" would mean the user cancelled."""
    pick = _pending(_FakeHandle(2, ["C:/pics/a.png"]))
    assert pick.result() is None
    assert pick.result() is None
    assert pick.result() == "C:/pics/a.png"


def test_a_cancelled_dialog_is_distinct_from_a_pending_one():
    pick = _pending(_FakeHandle(0, []))
    assert pick.result() == ""


def test_the_result_is_taken_only_once():
    """The caller polls every frame; a second read must not re-apply it."""
    pick = _pending(_FakeHandle(0, ["C:/pics/a.png"]))
    assert pick.result() == "C:/pics/a.png"
    assert pick.result() == ""


def test_a_missing_backend_is_not_an_error():
    """available() is False and a pick resolves immediately to nothing, so a
    typed path stays the way through."""
    pick = _pending(None)
    assert pick.result() == ""


def test_folder_of_reopens_where_the_last_choice_was(tmp_path):
    target = tmp_path / "pic.png"
    target.write_bytes(b"")
    assert file_picker.folder_of(str(target)) == str(tmp_path)
    assert file_picker.folder_of("") == ""


def test_an_absolute_shader_path_resolves(tmp_path):
    """The browser hands back a full path, which is not in any known folder."""
    frag = tmp_path / "mine.frag"
    frag.write_text("void main(){}")
    assert field_sources.resolve_shader_path(str(frag)) == Path(str(frag))
    assert field_sources.resolve_shader_path(str(tmp_path / "gone.frag")) is None
    assert field_sources.resolve_shader_path("") is None
