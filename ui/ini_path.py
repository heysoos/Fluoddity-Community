"""Where Dear ImGui persists the window layout.

One function, so there is one thing for the test suite to switch off. The
suite must never touch the real file: ImGui loads it in create_context and
writes it back in destroy_context, so a test run would both consume the user's
layout and leave its own behind - which once made an unrelated test fail with
no code change. See tests/conftest.py.
"""
from __future__ import annotations

from utilities.paths import get_imgui_ini_path

_suppressed = False


def suppress() -> None:
    """Stop install() from pointing ImGui at anything. Not reversible."""
    global _suppressed
    _suppressed = True


def is_suppressed() -> bool:
    return _suppressed


def install(io) -> str | None:
    """Point `io` at the user's layout file. -> the path, or None if suppressed.

    Only ever sets the filename; reading it back off a null segfaults.
    """
    if _suppressed:
        return None
    path = get_imgui_ini_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    io.set_ini_filename(str(path))
    return str(path)
