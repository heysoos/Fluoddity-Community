"""Native Open-file dialogs, asked for on one frame and collected on a later one.

The dialog runs in its own process, so `ready()` must be polled rather than
waited on: blocking the frame loop for as long as someone browses would freeze
the sim and the UI behind it.

Optional by design. `available()` is False where the backend is missing, and
every caller keeps a typed path as the way through - a picker is a convenience
over a control that must work without it.
"""
from __future__ import annotations

from pathlib import Path

_pfd = None
_checked = False

IMAGE_FILTER = ["Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tga *.webp",
                "All files", "*"]
SHADER_FILTER = ["Fragment shaders", "*.frag *.glsl", "All files", "*"]


def _backend():
    """Import lazily: the dialog module is not needed to draw a window."""
    global _pfd, _checked
    if not _checked:
        _checked = True
        try:
            from imgui_bundle import portable_file_dialogs
            _pfd = portable_file_dialogs
        except Exception:
            _pfd = None
    return _pfd


def available() -> bool:
    return _backend() is not None


class PendingPick:
    """One open dialog. Poll `result()` every frame until it is not None."""

    def __init__(self, title: str, filters: list[str], start_dir: str = ""):
        pfd = _backend()
        self._handle = None
        if pfd is None:
            return
        try:
            self._handle = pfd.open_file(title, start_dir, filters)
        except Exception:
            self._handle = None

    def result(self):
        """-> a path once chosen, "" if cancelled, None while still open."""
        if self._handle is None:
            return ""
        if not self._handle.ready(0):
            return None
        chosen = self._handle.result()
        self._handle = None
        return str(chosen[0]) if chosen else ""


def open_image(start_dir: str = "") -> PendingPick:
    return PendingPick("Choose an image", IMAGE_FILTER, start_dir)


def open_shader(start_dir: str = "") -> PendingPick:
    return PendingPick("Choose a shader", SHADER_FILTER, start_dir)


def folder_of(path: str) -> str:
    """Where a dialog should reopen, given what was chosen last."""
    if not path:
        return ""
    parent = Path(path).expanduser().parent
    return str(parent) if parent.is_dir() else ""
