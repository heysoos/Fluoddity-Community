"""Ticking Explore off must not rescore the archive on the frame loop.

The closing flush pays rescore_all(), which is O(n^2): measured 1.8 s at
11514 entries. The archive stays open when Explore stops, so that flush is
not a closing one; quitting and switching archive still pay it. See CLAUDE.md.
"""
from main import App


class _Archive:
    def __init__(self):
        self.calls = []

    def maybe_flush(self, every=200, force=False, closing=False):
        self.calls.append({"force": force, "closing": closing})
        return True


class _Svc:
    def __init__(self):
        self.driver = "explore"
        self.paused = 0

    def pause(self):
        self.paused += 1


def _stub():
    app = App.__new__(App)
    app.archive = _Archive()
    app.auto_service = _Svc()
    app.prompt_driver = "prompt"
    app._undo_auto_overrides = lambda ui: None
    return app


def test_leaving_explore_flushes_without_a_closing_rescore():
    app = _stub()
    app._leave_explore(None)
    assert app.archive.calls == [{"force": True, "closing": False}]


def test_leaving_explore_still_pauses_and_hands_the_driver_back():
    app = _stub()
    app._leave_explore(None)
    assert app.auto_service.paused == 1
    assert app.auto_service.driver == "prompt"
