"""Only the frame loop can switch brain, so this is where a layout move lands.

App owns the archive, the sim and the tournament; the driver owns none of them.
The move therefore reaches the app as a ONE-SHOT, in the frame the generation
that proposed it was scored.
"""
from __future__ import annotations

import types

from services.brains import REGISTRY, default_layout

FOURIER = default_layout()
GABOR = REGISTRY["gabor"].layout_from_settings({})


class _Drv:
    def __init__(self, requested=None):
        self.requested_layout = requested
        self.began = 0

    def begin_moved_expedition(self):
        self.began += 1
        return True


class _App:
    """Just enough App to drive the method under test."""

    def __init__(self, drv):
        import main

        self.imgep_driver = drv
        self.applied = []
        self._apply_requested_layout = types.MethodType(
            main.App._apply_requested_layout, self)

    def _apply_brain_layout(self, layout, ui_state, *, keep_running=False):
        self.applied.append((layout, keep_running))
        return True


def _ui():
    from state.archive_state import ArchiveState
    from state.brain_state import BrainState

    return types.SimpleNamespace(archive=ArchiveState(), brain=BrainState())


def test_nothing_happens_without_a_request():
    app = _App(_Drv())
    assert app._apply_requested_layout(_ui()) is False
    assert not app.applied


def test_no_driver_at_all_is_not_an_error():
    """Manual and Auto (Prompt) modes never build one."""
    app = _App(None)
    assert app._apply_requested_layout(_ui()) is False


def test_a_request_is_applied_without_stopping_the_search():
    """The search is moving its own space on purpose. Stopping it here would
    make a layout move cost the whole run."""
    drv = _Drv(GABOR)
    app, ui = _App(drv), _ui()
    assert app._apply_requested_layout(ui) is True
    assert app.applied == [(GABOR, True)]
    assert drv.requested_layout is None
    assert drv.began == 1


def test_the_brain_window_moves_with_the_sim():
    """_handle_brain_layout applies ui_state.brain every frame, so a switch
    that moves only the sim is undone by the next one."""
    drv = _Drv(GABOR)
    app, ui = _App(drv), _ui()
    app._apply_requested_layout(ui)
    assert ui.brain.modality == "gabor"


def test_the_request_is_consumed_even_if_the_switch_is_refused():
    """Left standing it would be re-applied every generation forever."""

    class Refuse(_App):
        def _apply_brain_layout(self, layout, ui_state, *, keep_running=False):
            self.applied.append((layout, keep_running))
            return False

    drv = _Drv(GABOR)
    app, ui = Refuse(drv), _ui()
    app._apply_requested_layout(ui)
    assert drv.requested_layout is None


def test_the_expedition_is_begun_after_the_switch_not_before():
    """A genome of the child's width cannot be optimised under the parent's
    spec, which is the whole reason a move is two stages."""
    order = []

    class Ordered(_App):
        def _apply_brain_layout(self, layout, ui_state, *, keep_running=False):
            order.append("apply")
            return True

    class Drv(_Drv):
        def begin_moved_expedition(self):
            order.append("begin")
            return True

    app = Ordered(Drv(GABOR))
    app._apply_requested_layout(_ui())
    assert order == ["apply", "begin"]
